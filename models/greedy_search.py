import torch


def rnnt_greedy_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    n_steps: int = 64,
):
    """
    Args:
        encoder_out: B=1, T, E
        encoder_out_lens: B=1
    """
    # fake padding
    padding = torch.zeros(1, 1).to(encoder_out.device)
    # sos
    pred_input_step = torch.tensor([model.blank]).reshape(1, 1).to(encoder_out.device)  # [B=1, T=1]
    cache = model.predictor.init_state(1, device=encoder_out.device)  # List[torch.Tensor[num_layer, B=1, hidden_dim]]
    new_cache = []
    t = 0
    hyps = []
    emos = []
    prev_out_nblk = True
    pred_out_step = None
    per_frame_max_noblk = n_steps
    per_frame_noblk = 0
    while t < encoder_out_lens:
        encoder_out_step = encoder_out[:, t:t + 1, :]  # [1, 1, E]
        if prev_out_nblk:
            pred_out_step, new_cache = model.predictor.forward_step(pred_input_step, padding, cache)  # [1, 1, P]

        joint_out_step = model.joint(encoder_out_step, pred_out_step)  # [1,1,1,v]
        joint_out_probs = joint_out_step.log_softmax(dim=-1)

        joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # []
        if joint_out_max != model.blank:
            hyps.append(joint_out_max.item())
            prev_out_nblk = True
            per_frame_noblk = per_frame_noblk + 1
            pred_input_step = joint_out_max.reshape(1, 1)
            # state_m, state_c =  clstate_out_m, state_out_c
            cache = new_cache

        if joint_out_max == model.blank or per_frame_noblk >= per_frame_max_noblk:
            if joint_out_max == model.blank:
                prev_out_nblk = False
            if model.joint_weight:
                joint_out_step_emotion = model.joint_emotion(encoder_out_step, pred_out_step)  # [1,1,1,4]
                joint_out_emo_probs = joint_out_step_emotion.log_softmax(dim=-1)
                joint_out_emo_max = joint_out_emo_probs.argmax(dim=-1).squeeze()
                emos.append(joint_out_emo_max.item())
            elif model.head_weight:
                head_out_step_emotion = model.head_emotion(encoder_out_step, torch.ones(1, device=encoder_out.device), pred_out_step, torch.ones(1, device=encoder_out.device))  # [1,4]
                head_out_emo_probs = head_out_step_emotion.log_softmax(dim=-1)
                head_out_emo_max = head_out_emo_probs.argmax(dim=-1).squeeze()
                emos.append(head_out_emo_max.item())
            # TODO(Mddct): make t in chunk for streamming
            # or t should't be too lang to predict none blank
            t = t + 1
            per_frame_noblk = 0

    return hyps, emos


def frnnt_greedy_search(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    encoder_emo_out: torch.Tensor = None,
    n_steps: int = 64,
    threshold: float = 0.5
):
    """
    Args:
        encoder_out: B=1, T, E
        encoder_emo_out: B=1, T, E
        encoder_out_lens: B=1
    """
    # fake padding
    padding = torch.zeros(1, 1).to(encoder_out.device)
    # sos, note pred_input_step is shared by predictor_blank and predictor_vocab
    pred_input_step = torch.tensor([model.blank]).reshape(1, 1).to(encoder_out.device)  # [B=1, T=1]
    cache_blank = model.predictor_blank.init_state(1, device=encoder_out.device)  # List[torch.Tensor[num_layer, B=1, hidden_dim]]
    cache_vocab = model.predictor_vocab.init_state(1, device=encoder_out.device)
    newb_cache = []
    newv_cache = []
    t = 0
    hyps = []
    emos = []
    prev_out_nblk = True
    pred_out_step_blank = None
    pred_out_step_vocab = None
    per_frame_max_noblk = n_steps
    per_frame_noblk = 0
    while t < encoder_out_lens:

        # audio at now time step
        encoder_out_step = encoder_out[:, t:t + 1, :]  # [1, 1, E]
        encoder_emo_out_step = encoder_emo_out[:, t:t + 1, :] if encoder_emo_out is not None else encoder_out[:, t:t + 1, :]

        # text with blank at now time step
        if prev_out_nblk:
            pred_out_step_blank, newb_cache = model.predictor_blank.forward_step(pred_input_step, padding, cache_blank)  # [1, 1, P]
            pred_out_step_vocab, newv_cache = model.predictor_vocab.forward_step(pred_input_step, padding, cache_vocab)
        joint_out_step_blank = model.joint_blank(encoder_emo_out_step, pred_out_step_blank)  # [1,1,1,1]
        joint_out_step_vocab = model.joint_vocab(encoder_out_step, pred_out_step_vocab)  # [1,1,1,v-1]
        joint_out_step = torch.cat((joint_out_step_blank, joint_out_step_vocab), dim=-1)
        joint_out_probs = joint_out_step.log_softmax(dim=-1)
        joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # []

        # emotion at current time step

        if joint_out_max != model.blank:
            hyps.append(joint_out_max.item())
            prev_out_nblk = True
            per_frame_noblk = per_frame_noblk + 1
            pred_input_step = joint_out_max.reshape(1, 1)
            cache_blank = newb_cache
            cache_vocab = newv_cache

        if joint_out_max == model.blank or per_frame_noblk >= per_frame_max_noblk:
            if joint_out_max == model.blank:
                prev_out_nblk = False
            if model.joint_weight:
                joint_out_step_emotion = model.joint_emotion(encoder_emo_out_step, pred_out_step_blank)  # [1,1,1,4]
                joint_out_emo_probs = joint_out_step_emotion.log_softmax(dim=-1)
                joint_out_emo_max = joint_out_emo_probs.argmax(dim=-1).squeeze()
                emos.append(joint_out_emo_max.item())
                
            elif model.head_weight:
                head_out_step_emotion = model.head_emotion(encoder_emo_out_step, torch.ones(1, device=encoder_out.device), pred_out_step_blank, torch.ones(1, device=encoder_out.device))  # [1,4]
                head_out_emo_probs = head_out_step_emotion.log_softmax(dim=-1)
                head_out_emo_max = head_out_emo_probs.argmax(dim=-1).squeeze()
                emos.append(head_out_emo_max.item())
                
            t = t + 1
            per_frame_noblk = 0

    return hyps, emos