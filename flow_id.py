import torch
from diffusers.image_processor import VaeImageProcessor
import torch.nn.functional as F
import torch
import bitsandbytes as bnb
from copy import deepcopy
import matplotlib.pyplot as plt
import gc
from sklearn.cluster import KMeans
from torchvision.transforms.functional import gaussian_blur
from diffusers import StableDiffusion3Pipeline
from flow_id_sd3_dit import FlowIDSD3Transformer2DModel
from tqdm import tqdm

def logit_sampler(mean_logit : float = 0, std_logit : float = 1, batch_size : int = 10, device : str = "cuda", dtype = torch.bfloat16):
    u = torch.normal(mean=mean_logit, std=std_logit, size=(batch_size,), device=device, dtype = dtype)
    u = torch.nn.functional.sigmoid(u)
    return u

def kmeans_mask(map_tensor):
    km = KMeans(2, n_init=1, random_state=42).fit(map_tensor.cpu().numpy().reshape(-1,1))
    mask = torch.from_numpy(km.labels_.reshape(map_tensor.shape)).float()
    return mask if km.cluster_centers_[0] < km.cluster_centers_[1] else 1-mask

class FlowID:
    def __init__(self):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            transformer = None,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        transformer = FlowIDSD3Transformer2DModel.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="transformer",
            torch_dtype = torch.bfloat16
        )
        self.transformer_copy = deepcopy(transformer)
        self.pipe.transformer = transformer.to("cuda")
        

    def _get_indexes(self,prompt,concept):
        def find_indexes(prompt_token_list,indexes,concept):
            concept = concept.replace(" ","").replace(",","")
            word = "" 
            for i in indexes:
                word += prompt_token_list[i]
            word = word.replace(" ","").replace(",","")
            if  word == concept:
                return indexes
            elif word in concept:
                indexes.append(indexes[-1]+1)
                return find_indexes(prompt_token_list,indexes,concept)
            elif word not in concept and prompt_token_list[indexes[-1]] not in concept:
                indexes = [indexes[-1]+1]
                return find_indexes(prompt_token_list,indexes,concept)
            elif word not in concept and prompt_token_list[indexes[-1]] in concept:
                indexes = [indexes[-1]]
            return find_indexes(prompt_token_list,indexes,concept)

        tokens = self.pipe.tokenizer_3.encode(prompt)
        prompt_token_list = [self.pipe.tokenizer_3.decode([token]) for token in tokens]

        idxs = find_indexes(prompt_token_list,[0],concept.replace(" ","").replace(",","").replace(".",""))
        
        return idxs

    @torch.no_grad()
    def _prepare_z0(self,img):
        self.pipe.vae.cuda()
        image_processor = VaeImageProcessor(vae_scale_factor=8, vae_latent_channels=16)
        img = image_processor.preprocess(img)
        image = img.to(device="cuda", dtype = torch.bfloat16)
        init_latents = self.pipe.vae.encode(image).latent_dist.sample(None)
        init_latents = (init_latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        z0 = torch.cat([init_latents], dim=0)
        self.pipe.vae.cpu()
        return z0
        
    @torch.no_grad()
    def _prepare_prompt_embeds(self,prompt,neg_prompt=""):
        self.pipe.text_encoder.cuda()
        self.pipe.text_encoder_2.cuda()
        self.pipe.text_encoder_3.cuda()
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                negative_prompt=neg_prompt,
                negative_prompt_2=neg_prompt,
                negative_prompt_3=neg_prompt,
                device="cuda",
            ) 

        prompt_embeds = torch.cat([negative_prompt_embeds,prompt_embeds],dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds,pooled_prompt_embeds],dim=0)

        self.pipe.text_encoder.cpu()
        self.pipe.text_encoder_2.cpu()
        self.pipe.text_encoder_3.cpu()

        return prompt_embeds,pooled_prompt_embeds

    def _compute_mask(
            self,
            map,
            kernel_size,
            do_gaussian_blur
        ):
        map = F.interpolate(map.unsqueeze(0).unsqueeze(0), size=(128,128),mode="bilinear").squeeze(0).squeeze(0)
        mask = kmeans_mask(map)
        if kernel_size != 0:
            mask = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0),kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze()
        if do_gaussian_blur:
            mask = gaussian_blur(mask.unsqueeze(0).unsqueeze(0),kernel_size=9,sigma=9).squeeze() #ajouté
        return mask
    
    @torch.no_grad()
    def precompute_mask(
        self,
        image,
        num_samples,
        prompt,
        concept,
        show_attention=True,
        show_mask=True,
        do_gaussian_blur=True,
        kernel_size=1
    ):
        self.z0 = self._prepare_z0(image)
        prompt_embeds,pooled_prompt_embeds = self._prepare_prompt_embeds(prompt)
        indexes = self._get_indexes(prompt,concept)
        avg_map = torch.zeros((64,64))
        for _ in tqdm(range(num_samples)):
            sigma = torch.rand(1,device="cuda",dtype=torch.bfloat16)
            noise = torch.randn_like(self.z0)
            timestep = (sigma * 1000).float()
            model_input = (1 - sigma) * self.z0 + sigma * noise
            map = self.pipe.transformer(
                hidden_states = model_input,
                timestep = timestep,
                encoder_hidden_states = prompt_embeds,
                pooled_projections = pooled_prompt_embeds,
                concept_indexes = indexes
            )[1]
            avg_map += map

        if show_attention:
            plt.imshow(avg_map)
            plt.show()
        
        if show_mask:
            mask = self._compute_mask(
                avg_map,
                kernel_size=kernel_size,
                do_gaussian_blur=do_gaussian_blur
            )
            plt.imshow(mask.float().cpu())
            plt.show()

        return mask
    
    @torch.no_grad()
    def ft_invert(
        self,
        img,
        prompt,
        num_train_steps = 50,
        simulated_batch_size = 10,
        batch_size = 1,
        num_inversion_steps = 50,
        lr = 5e-5
    ):
        prompt_embeds,pooled_prompt_embeds = self._prepare_prompt_embeds(
            prompt
        )
        self.z0 = self._prepare_z0(img)

        fine_tune = num_train_steps > 0

        if fine_tune:
            self._reset_transformer()
            optimizer = bnb.optim.Adam8bit(self.pipe.transformer.parameters(),lr=lr)
            self.pipe.transformer.requires_grad_(True)
            optimizer.zero_grad()
            z0 = torch.cat([self.z0]*batch_size)
            with torch.enable_grad():
                for i in tqdm(range(num_train_steps*int(simulated_batch_size/batch_size))):
                    noise = torch.randn_like(z0)
                    sigmas = logit_sampler(batch_size = batch_size, dtype = torch.bfloat16)
                    t=torch.tensor(sigmas*1000,device="cuda", dtype = torch.bfloat16)
                    sigmas = sigmas.view(batch_size,1,1,1)
                    noised_latents = (1.0 - sigmas) * z0 + sigmas * noise
                    target_vf = noise - z0
                    predicted_vf = self.pipe.transformer(
                            hidden_states = noised_latents,
                            timestep = t,
                            encoder_hidden_states = prompt_embeds[1],
                            pooled_projections = pooled_prompt_embeds[1]
                    )[0]
                    loss = torch.nn.functional.mse_loss(predicted_vf, target_vf)
                    loss.backward()

                    if (i+1) % simulated_batch_size == 0:
                        optimizer.step()
                        optimizer.zero_grad()

        zts = [self.z0]
        self.num_inversion_steps = num_inversion_steps
        self.pipe.scheduler.set_timesteps(self.num_inversion_steps)
        self.timesteps = self.pipe.scheduler.timesteps.cuda()
        self.sigmas = self.pipe.scheduler.sigmas
        self.inversion_timesteps = torch.cat([torch.tensor([0.0],device="cuda"),self.timesteps.flip(0)[:-1]])
        self.inversion_sigmas = self.sigmas.flip(0)
        latent = self.z0.clone()

        for i,t in tqdm(enumerate(self.inversion_timesteps)):
            model_input = latent
            timestep = t.expand(model_input.shape[0])
            vf_pred = self.pipe.transformer(
                hidden_states = model_input,
                timestep = timestep,
                encoder_hidden_states = prompt_embeds[1],
                pooled_projections = pooled_prompt_embeds[1]
            )[0]
            latent += (self.inversion_sigmas[i+1]-self.inversion_sigmas[i])*vf_pred
            zts.append(latent.detach().clone())

        self.zts_ref = zts[::-1] 
        self.original_zts = zts[::-1] 
        self.prev_prompt = prompt

    @torch.no_grad()
    def edit(
            self,
            prompt,
            neg_prompt=None,
            concept=None,
            num_skipped_steps=50,
            kernel_size=0,
            show_mask=True,
            idx_t_mask = None,
            do_gaussian_blur = True,
            pasting_cutoff = 0,
            cfg = 7,
            mask = None,
            output_mask = False
        ):
        do_masking = concept is not None
        if (concept is not None) and (concept not in prompt) and (concept in self.prev_prompt):
            remove = True
            preserve = False
        elif (concept is not None) and (concept in prompt) and (concept in self.prev_prompt):
            preserve = True
            remove = False
        else:
            remove = False
            preserve = False

        if idx_t_mask is None:
            idx_t_mask = int(self.num_inversion_steps/5)
        if remove is not True and  idx_t_mask <= num_skipped_steps:
            idx_t_mask = num_skipped_steps 

        do_cfg = cfg > 1

        self.last_num_skipped_steps = num_skipped_steps
        prompt_embeds,pooled_prompt_embeds = self._prepare_prompt_embeds(prompt,neg_prompt) 
        timesteps = self.timesteps[num_skipped_steps:]
        sigmas = self.sigmas[num_skipped_steps:]
        zts_ref = self.zts_ref[num_skipped_steps:]        
        zT = zts_ref[0]

        if do_masking:
            if mask is None:
                if remove:
                    zt_for_mask = self.zts_ref[idx_t_mask]
                    t = self.timesteps[idx_t_mask]
                    prompt_embeds_for_mask, pooled_prompt_embeds_for_mask = self._prepare_prompt_embeds(self.prev_prompt,neg_prompt) 
                    indexes = self._get_indexes(self.prev_prompt,concept)
                else:
                    latent = zT.clone()
                    for i,t in tqdm(enumerate(timesteps[:idx_t_mask-num_skipped_steps])):
                        model_input = torch.cat([latent]*2) if do_cfg else latent
                        timestep = t.expand(model_input.shape[0])
                        vf_pred = self.pipe.transformer(
                                hidden_states = model_input,
                                timestep = timestep,
                                encoder_hidden_states = prompt_embeds,
                                pooled_projections = pooled_prompt_embeds
                        )[0]
                        if do_cfg:
                            vf_pred_neg, vf_pred_prior = vf_pred.chunk(2)
                            vf_pred = vf_pred_neg + cfg*(vf_pred_prior-vf_pred_neg)
                        latent += (sigmas[i+1]-sigmas[i])*vf_pred
                    t = timesteps[idx_t_mask-num_skipped_steps]
                    zt_for_mask = latent.clone()
                    prompt_embeds_for_mask, pooled_prompt_embeds_for_mask = prompt_embeds, pooled_prompt_embeds 
                    indexes = self._get_indexes(prompt,concept)
                
                model_input = torch.cat([zt_for_mask]*2)
                timestep = t.expand(model_input.shape[0])
                map = self.pipe.transformer(
                        hidden_states = model_input,
                        timestep = timestep,
                        encoder_hidden_states = prompt_embeds_for_mask,
                        pooled_projections = pooled_prompt_embeds_for_mask,
                        concept_indexes = indexes
                )[1]
                mask = self._compute_mask(
                    map,
                    kernel_size=kernel_size,
                    do_gaussian_blur=do_gaussian_blur
                )
                del prompt_embeds_for_mask, pooled_prompt_embeds_for_mask
            if show_mask:
                plt.imshow(mask.float().cpu())
                plt.show()

        new_zts_ref = [zT]
        latent = zT.clone()
        for i,t in tqdm(enumerate(timesteps)):
            model_input = torch.cat([latent]*2)
            timestep = t.expand(model_input.shape[0])
            vf_pred = self.pipe.transformer(
                    hidden_states = model_input,
                    timestep = timestep,
                    encoder_hidden_states = prompt_embeds,
                    pooled_projections = pooled_prompt_embeds
            )[0]
            vf_pred_neg, vf_pred_prior = vf_pred.chunk(2)
            vf_pred = vf_pred_neg + 7*(vf_pred_prior-vf_pred_neg)
            latent += (sigmas[i+1]-sigmas[i])*vf_pred
            if (i < self.num_inversion_steps - pasting_cutoff - num_skipped_steps) and do_masking:
                if remove:
                    latent = latent * mask.cuda().to(torch.bfloat16) + zts_ref[i+1] * (1 - mask.cuda().to(torch.bfloat16))
                elif preserve:
                    latent = latent * (1 - mask.cuda().to(torch.bfloat16)) + zts_ref[i+1] * mask.cuda().to(torch.bfloat16)
            new_zts_ref.append(latent)

        self.tmp_zts = new_zts_ref
        self.tmp_prompt = prompt

        self.pipe.vae.cuda()
        image_processor = VaeImageProcessor(vae_scale_factor=8, vae_latent_channels=16)
        latents_rescaled = (latent / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents_rescaled, return_dict=False)[0]
        edited = image_processor.postprocess(image, output_type='pil')[0] 
        self.pipe.vae.cpu()

        if output_mask:
            return (edited,mask)
        else:
            return (edited,)

    def validate_edit(self):
        self.zts_ref = self.original_zts[:self.last_num_skipped_steps]+self.tmp_zts
        self.prev_prompt = self.tmp_prompt

    def _reset_transformer(self):
        del self.pipe.transformer
        torch.cuda.empty_cache()
        gc.collect()
        self.pipe.transformer = deepcopy(self.transformer_copy).to("cuda")
