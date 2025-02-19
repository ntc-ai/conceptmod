from omegaconf import OmegaConf
from collections import defaultdict

from safetensors import safe_open
from datasets import load_dataset

import sys; sys.path.append('.')
import torch
from torch import autocast
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
#import ImageReward as reward
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import random
import glob
import re
import shutil
import pdb
import argparse
from convertModels import savemodelDiffusers
import torchvision.transforms.functional as F


import time
from contextlib import nullcontext
from PIL import Image

def compare(img1, img2):
    return ((img1-img2)**2).mean()

def score_tensor(self, prompt, image):
    # text encode
    text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
    # image encode

    image = image.permute(2, 0, 1)

    # Define the transformation pipeline
    resize_size = 224
    normalize_mean = (0.48145466, 0.4578275, 0.40821073)
    normalize_std = (0.26862954, 0.26130258, 0.27577711)

    # Apply the transformations
    image_resized = F.resize(image, (resize_size, resize_size))
    image_normalized = F.normalize(image_resized, mean=normalize_mean, std=normalize_std)
    image_normalized = image_normalized.to(torch.float32)

    image_embeds = self.blip.visual_encoder(image_normalized.unsqueeze(0))
    # text encode cross attention with image
    image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
    text_output = self.blip.text_encoder(text_input.input_ids,
                                            attention_mask = text_input.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,
                                            return_dict = True,
                                        )
    txt_features = text_output.last_hidden_state[:,0,:].float() # (feature_dim)
    rewards = self.mlp(txt_features)
    rewards = (rewards - self.mean) / self.std
    return rewards

#reward.ImageReward.score_tensor = score_tensor



model = None
def score_image(prompt, img):
    global model
    if model is None:
        model = reward.load("ImageReward-v1.0").to("cuda:0")
    return model.score_tensor(prompt, img)

def proximal_operator_l1(tensor, lmbd):
    return torch.sign(tensor) * torch.relu(torch.abs(tensor) - lmbd)

# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    tensors = {}
    mPath=ckpt
    if "safetensors" in mPath:
        with safe_open(mPath, framework="pt", device="cpu") as f:
           for key in f.keys():
               tensors[key] = f.get_tensor(key).cpu()

        #global_step = pl_sd["global_step"]
        sd = tensors#pl_sd["state_dict"]
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd#["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)

##################### ESD Functions
def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler

def parse_input_string(input_str):
    params = {
        "alpha": 1.0,  # Default alpha value
    }

    # Split the input string by ':' to get the concepts and parameters
    parts = input_str.split(':')

    # Set the concept
    params["concept"] = parts[0]

    # Iterate through the remaining parts to parse parameters
    for part in parts[1:]:
        # Check if the parameter has a '=' sign, indicating a key-value pair
        if '=' in part:
            key, value = part.split('=', 1)
            params[key] = float(value)
        else:
            negative=False
            if part.startswith("--"):
                negative=True
                part = part[1:]
            # If it's just a value, assume it's the alpha value
            params["alpha"] = float(part)
            if negative:
                params["alpha"]=-params["alpha"]

    return params

def sample_image(name, model, sampler, sample_start_code, sample_emb, step, ddim_steps, save=False):
    start_code = sample_start_code
    device = sample_start_code.device

    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                tic = time.time()
                uc = None
                uc = model.get_learned_conditioning([""])
                c = sample_emb
                shape = [4, 64, 64]
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=1,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=7.5,
                                                 unconditional_conditioning=uc,
                                                 eta=0.0,
                                                 x_T=start_code)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                if not save:
                    return x_samples_ddim.permute(0, 2, 3, 1).squeeze(0)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
                x_sample = x_samples_ddim

                x_sample = 255. * x_sample
                x_sample = x_sample.astype(np.uint8)
                img = Image.fromarray(x_sample)
                img.save(f"{name}/{step:05}.png")
                return x_samples_ddim

def process_rule(rule):
    if '~' not in rule:
        return [rule]

    prefix, rest = rule.split('~')
    target, lambda_value = (rest.split(':') + ['0.1'])[:2]
    lambda_value = float(lambda_value)

    new_rules = [
        f"{target}++:{2 * lambda_value}",
        f"{prefix}={target}:{4 * lambda_value}",
        f"{target}%{prefix}:-{lambda_value}"
    ]

    return new_rules


def move_towards(target_model, source_model, alpha=0.3):
    target_state = target_model.state_dict()
    source_state = source_model.state_dict()

    for key in target_state:
        target_state[key] = target_state[key] * (1 - alpha) + source_state[key] * alpha

    target_model.load_state_dict(target_state)


def train_esd(prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, diffusers_config_path, devices, seperator=None, image_size=512, ddim_steps=50, sample_prompt=None, accumulation_steps=1, randomly_pull_prompts=False, merge_speed=0.05, merge_every=0):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (ESD-x, ESD-u, full, selfattn).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.
    merge_speed: float, optional
        How quickly to merge from old model to new model. The default is 0.05
    merge_every: int, optional
        How many steps before merging. The default is 0 (off)

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')

    if seperator is not None:
        rules = prompt.split(seperator)
        rules = [word.strip() for word in rules]
    else:
        rules = [prompt]
    ddim_eta = 0
    # MODEL TRAINING SETUP

    model_orig, sampler_orig, model, sampler = get_models(config_path, ckpt_path, devices)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == 'full':
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                print(name)
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    print(name)
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    print(name)
                    parameters.append(param)
    # set model to train
    model.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)


    criteria = torch.nn.MSELoss()

    history = []

    name = f'compvis-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}'
    name = name[0:50]
    # TRAINING CODE
    pbar = tqdm(range(iterations))

    if sample_prompt is not None:
        sample_start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
        sample_emb = model.get_learned_conditioning([sample_prompt])

        os.makedirs("samples/"+name, exist_ok=True)
        sample_image("samples/"+name, model, sampler, sample_start_code, sample_emb, 0, ddim_steps, save=True)
    accumulation_counter=0

    dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts")
    orules = list(rules)
    opt.zero_grad()
    cache = {}

    # Define a function to generate a unique key for caching
    def generate_cache_key(model, emb_text, grad):
        return (id(model), emb_text, grad)

    # Define a function to apply the model and cache the results
    def apply_model_cache(model, emb_text, z, t_enc_ddpm, emb, cache, grad=False):
        key = generate_cache_key(model, emb_text, grad)
        if grad or key not in cache:
            cache[key] = model.apply_model(z, t_enc_ddpm, emb)
        return cache[key]

    for i in pbar:
        rules = list(orules)
        random_prompt = random.choice(dataset['train']["Prompt"]).replace(":","-").replace("%", " percent ").replace("="," equals ")
        rules = [item for rule in rules for item in process_rule(rule)]
        print(len(rules), "--", rules)

        if args.merge_every != 0 and (i+1) % args.merge_every == 0:
            print("Moving towards")
            opt.state = defaultdict(dict)
            move_towards(model_orig, model, alpha=merge_speed)

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])


        cache.clear()

        rule_losses = []
        for rule_params in rules:
            rule_index = rules.index(rule_params)
            rule_params = rule_params.replace("{random_prompt}", random_prompt)
            rule_obj = parse_input_string(rule_params)
            rule = rule_obj['concept']
            if rule == '':
                continue


            if '=' in rule:
                # Handle the concept replacement case (original=target)
                concepts = rule.split('=')
                trainable_concept = concepts[0]
                target_concept = concepts[1]

                # Get text embeddings for unconditional and conditional prompts
                emb_0 = model.get_learned_conditioning([''])
                emb_o = model.get_learned_conditioning([trainable_concept])
                emb_t = model.get_learned_conditioning([target_concept])

                assert target_concept != ""

                with torch.no_grad():
                    # Generate an image with the target concept from ESD model
                    z = quick_sample_till_t(emb_t.to(devices[0]), start_guidance, start_code, int(t_enc))

                    # Get conditional and unconditional scores from frozen model at time step t and image z
                    e_0 = apply_model_cache(model_orig, '', z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]), cache)
                    e_o = apply_model_cache(model_orig, target_concept, z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_t.to(devices[1]), cache)


                # Get conditional scores from ESD model for the original concept
                e_t = apply_model_cache(model, trainable_concept, z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_o.to(devices[0]), cache, grad=True)

                # Compute the loss function for concept replacement
                loss_replacement = criteria(e_t.to(devices[0]), e_0.to(devices[0])) - (negative_guidance * (e_o.to(devices[0])- e_0.to(devices[0])))
                loss_rule = rule_obj['alpha']*loss_replacement.mean()
                rule_losses.append(loss_rule)

            elif '#' in rule:
                concepts = rule.split('#')
                trainable_concept = concepts[0]
                target_concept = concepts[1]

                emb_o = model.get_learned_conditioning([trainable_concept])
                emb_t = model.get_learned_conditioning([target_concept])

                with torch.no_grad():
                    z = quick_sample_till_t(emb_t.to(devices[0]), start_guidance, start_code, int(t_enc))
                    e_o = apply_model_cache(model_orig, target_concept, z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_t.to(devices[1]), cache)

                e_t = apply_model_cache(model, trainable_concept, z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_o.to(devices[0]), cache, grad=True)

                loss_replacement = criteria(e_t.to(devices[0]), e_o.to(devices[0]))
                loss_rule = rule_obj['alpha']*loss_replacement.mean()
                rule_losses.append(loss_rule)

            elif '^' in rule:
                concepts = rule.split('^')
                original_concept = concepts[0]
                target_concept = concepts[1]

                emb_t = model.get_learned_conditioning([target_concept])
                emb_o = model.get_learned_conditioning([original_concept])

                with torch.no_grad():
                    z = quick_sample_till_t(emb_t.to(devices[0]), start_guidance, start_code, int(t_enc))
                    e_o = apply_model_cache(model_orig, target_concept, z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_t.to(devices[1]), cache)

                e_t = apply_model_cache(model, original_concept, z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_o.to(devices[0]), cache, grad=True)


                assert False, "There were no gradients going through here so it's disabled."

                #img1 = sample_image("samples/"+name, model, sampler, start_code, emb_o, 0, ddim_steps, save=False)
                #img2 = sample_image("samples/"+name, model, sampler, start_code, emb_t, 0, ddim_steps, save=False)
                #loss_rule = rule_obj['alpha']* compare(img1, img2)
                #rule_losses.append(loss_rule)
                

            elif rule[0]==';':
                target_concept = rule[1:]

                emb_t = model.get_learned_conditioning([target_concept])

                e_t = apply_model_cache(model, original_concept, z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_t.to(devices[0]), cache, grad=True)

                assert False, "There were no gradients going through here so it's disabled."

                #img = sample_image("samples/"+name, model, sampler, start_code, emb_o, 0, ddim_steps, save=False)
                #loss_rule = -rule_obj['alpha']* score_image(target_concept, img)
                #rule_losses.append(loss_rule)




            elif rule[-2:] == '++':
                # Handle the concept insertion case (concept++)
                concept_to_insert = rule[:-2]
                guide = -negative_guidance
                _start_guidance = start_guidance

                # Get text embeddings for unconditional and conditional prompts
                emb_0 = model.get_learned_conditioning([''])
                emb_i = model.get_learned_conditioning([concept_to_insert])

                with torch.no_grad():
                    # Generate an image from ESD model
                    z = quick_sample_till_t(emb_0.to(devices[0]), _start_guidance, start_code, int(t_enc))

                    e_o = apply_model_cache(model_orig, concept_to_insert, z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_i.to(devices[1]), cache)
                    e_0 = apply_model_cache(model_orig, '', z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]), cache)
                    e_02 = apply_model_cache(model, '', z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]), cache)

                # Get conditional scores from ESD model for the concept to insert
                e_t = apply_model_cache(model, concept_to_insert, z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_i.to(devices[0]), cache, grad=True)
                if "guidance" in rule_obj:
                    guide=rule_obj["guidance"]
                target_diff = (e_o - e_0)**2*guide
                diff = (e_t-e_02.to(devices[0]))**2
                loss_i = torch.relu(target_diff - diff)
                loss_rule = rule_obj["alpha"]*loss_i.mean()
                rule_losses.append(loss_rule)

            elif rule[-2:] == '--':
                # Handle the concept insertion case (concept++)
                concept_to_insert = rule[:-2]
                guide = negative_guidance

                # Get text embeddings for unconditional and conditional prompts
                emb_0 = model.get_learned_conditioning([''])
                emb_i = model.get_learned_conditioning([concept_to_insert])

                with torch.no_grad():
                    # Generate an image from ESD model
                    z = quick_sample_till_t(emb_0.to(devices[0]), start_guidance, start_code, int(t_enc))

                    e_02 = apply_model_cache(model, '', z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]), cache)

                # Get conditional scores from ESD model for the concept to insert
                e_t = apply_model_cache(model, concept_to_insert, z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_i.to(devices[0]), cache, grad=True)

                loss_i = criteria(e_t.to(devices[0]), e_02.to(devices[0]))
                loss_rule = rule_obj["alpha"]*loss_i.mean()
                rule_losses.append(loss_rule)

            elif '%' in rule:
                # Handle the concept orthogonality case (concept1%concept2)
                concept1, concept2 = rule.split('%')
                # Get text embeddings for unconditional and conditional prompts
                emb_0 = model.get_learned_conditioning([''])

                emb_c1 = model.get_learned_conditioning([concept1])
                emb_c2 = model.get_learned_conditioning([concept2])

                with torch.no_grad():
                    # Generate an image from ESD model
                    z = quick_sample_till_t(emb_0.to(devices[0]), start_guidance, start_code, int(t_enc))

                    # Get conditional and unconditional scores from frozen model at time step t and image z
                    e_0 = apply_model_cache(model_orig, '', z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]), cache)
                    output_c1 = apply_model_cache(model_orig, concept1, z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_c1.to(devices[1]), cache)

                e_02 = apply_model_cache(model, '', z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]), cache)
                # Get conditional scores from ESD model for the two concepts
                output_c2 = apply_model_cache(model, concept2, z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_c2.to(devices[0]), cache, grad=True)
                diff_c1 = output_c1 - e_0
                diff_c2 = output_c2 - e_02

                diff_c1_flat = diff_c1.view(1, -1)
                diff_c2_flat = diff_c2.view(1, -1)
                # Normalize the output embeddings
                normalized_output_c1 = diff_c1_flat / (torch.norm(diff_c1_flat, dim=1, keepdim=True)+1e-12)
                normalized_output_c2 = diff_c2_flat / (torch.norm(diff_c2_flat, dim=1, keepdim=True)+1e-12)

                # Calculate the cosine similarity between the normalized output embeddings
                cosine_similarity = torch.abs(torch.dot(normalized_output_c1.view(-1).to(devices[0]), normalized_output_c2.view(-1).to(devices[0])))

                loss_rule = rule_obj["alpha"]*0.01*cosine_similarity.mean()
                rule_losses.append(loss_rule)

            else:
                assert False, "Unable to parse rule: "+rule

        debug = False
        if debug:
            grad_magnitudes = []
            for rule_index, rule_loss in enumerate(rule_losses):
                # Zero the gradients before calculating the gradients for the current rule_loss
                model.zero_grad()

                # Calculate gradients for the current rule_loss
                print("backprop", rule_index)
                print(rule_loss)
                rule_loss.backward(retain_graph=True)

                # Calculate the gradient magnitude for the current rule_loss
                grad_magnitude = 0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_magnitude += torch.norm(param.grad).item()
                grad_magnitudes.append(grad_magnitude)

                # Reapply the gradient from the total loss (for the next iteration)
                #loss.backward()
            model.zero_grad()

            total_grad_magnitude = sum(grad_magnitudes)
            percentage_contributions = [gm / total_grad_magnitude * 100 for gm in grad_magnitudes]
            for j, rule in enumerate(rules):
                print("{}: {:.2f}%".format(rule, percentage_contributions[j]))

        loss = sum(rule_losses)
        # Update weights to erase or reinforce the concept(s)
        loss.backward()
        for j, r in enumerate(rule_losses):
            print("{:.5f}".format(rule_losses[j].item()), rules[j])
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        accumulation_counter+=1
        if accumulation_counter % accumulation_steps == 0:
            opt.step()
            opt.zero_grad()
            if sample_prompt is not None:
                os.makedirs("samples/"+name, exist_ok=True)
                sample_image("samples/"+name, model, sampler, sample_start_code, sample_emb, (accumulation_counter//accumulation_steps), ddim_steps, save=True)

        # save checkpoint and loss curve
        if i == iterations-1 or (i+1) % 800 == 0:
            save_model(model, name, i-1, save_compvis=True, save_diffusers=False)

        if i % 100 == 0:
            save_history(losses, name, word_print)


    model.eval()

    save_model(model, name, None, save_compvis=True, save_diffusers=False, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    save_history(losses, name, word_print)

def save_model(model, name, num, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    # SAVE MODEL

#     PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'

    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.ckpt'
    else:
        path = f'{folder_path}/{name}.ckpt'
    print("Saved model to "+path)
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, device=device )

def save_history(losses, name, word_print):
    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3.0)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=-3.0)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=int, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='/sd-models/SDv1-5.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default="|")
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--accumulation_steps', help='gradient accumulation steps', type=int, required=False, default=2)
    parser.add_argument('--sample_prompt', help='will create training images with this phrase as SD trains. This requires running through SD and is slower.', type=str, required=False, default=None)
    parser.add_argument('--randomly_pull_prompts', help='pull unconditional towards "Gustavosta/Stable-Diffusion-Prompts".', type=bool, required=False, default=False)
    parser.add_argument('--merge_speed', help='Speed at which to merge the old model to the new model.', type=float, required=False, default=0.05)
    parser.add_argument('--merge_every', help='Step count before merging to new model. 0 is off', type=float, required=False, default=0)
    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    sample_prompt = args.sample_prompt
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    merge_speed = args.merge_speed
    merge_every = args.merge_every

    train_esd(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, sample_prompt=sample_prompt, accumulation_steps=args.accumulation_steps, randomly_pull_prompts=args.randomly_pull_prompts, merge_speed=merge_speed, merge_every=merge_every)
