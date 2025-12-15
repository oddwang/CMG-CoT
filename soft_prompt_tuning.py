import torch.nn.functional as F
from utils import *

def prompt_tuning(model, tokenizer, prompt, add_str="\nA: Let's think step by step.", soft_prompt_len=10, steps=200, tau=0.6, seed=1):
    device = "cuda"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False  # 冻结模型

    hidden_size = model.config.hidden_size
    embed_dtype = model.get_input_embeddings().weight.dtype

    prompt = prompt.replace('[/INST]', ' So, the answer is ([/INST]')

    split_index = prompt.rindex(add_str)

    prefix_text = prompt[:split_index]
    suffix_text = prompt[split_index:]

    prefix_tokens = tokenizer(prefix_text, add_special_tokens=False).input_ids
    suffix_tokens = tokenizer(suffix_text, add_special_tokens=False).input_ids

    insert_pos = len(prefix_tokens)
    input_ids = prefix_tokens + suffix_tokens
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    soft_prompt = (torch.randn(soft_prompt_len, hidden_size, device=device, dtype=torch.float32) * 0.001)
    soft_prompt.requires_grad_(True)

    optimizer = torch.optim.Adam([soft_prompt], lr=1e-3)
    target_list = [tokenizer(w, add_special_tokens=False).input_ids[0] for w in ["A", "B", "C", "D"]]

    has_optimized = False
    for step in range(steps):
        optimizer.zero_grad()

        with torch.no_grad():
            base_embeds = model.get_input_embeddings()(input_ids).to(embed_dtype)

        inputs_embeds = torch.cat([
            base_embeds[:, :insert_pos, :],
            soft_prompt.unsqueeze(0).to(embed_dtype),
            base_embeds[:, insert_pos:, :]
        ], dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(inputs_embeds=inputs_embeds)
            logits = outputs.logits[0, -1, :].float()
            logits = logits.clamp(-50, 50)
            probs = F.softmax(logits, dim=-1)
            eps = 1e-8
            p_list = torch.clamp(probs[target_list], eps, 1.0)
            max_prob = torch.max(p_list)
            loss = F.relu(max_prob - tau) ** 2

        if max_prob.item() <= tau:
            print(f"✅ Step {step:03d} | satisfied the predefined threshold (max_prob={max_prob.item():.4f} <= {tau})，stop optimization")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_([soft_prompt], max_norm=0.5)
        optimizer.step()

        if torch.isnan(soft_prompt).any() or torch.isinf(soft_prompt).any():
            with torch.no_grad():
                soft_prompt.data = torch.randn_like(soft_prompt) * 0.001
            continue

        if step % 5 == 0:
            avg_p = [round(p.item(), 5) for p in p_list]
            print(f"Step {step:03d} | Loss={loss.item():.6f} | probs={avg_p}")

    if step > 0:
        has_optimized = True
    return soft_prompt, has_optimized