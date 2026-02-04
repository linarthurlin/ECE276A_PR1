import torch

def quaternion_multiplication(q, p):
    qs, qv = q[..., 0:1], q[..., 1:]
    ps, pv = p[..., 0:1], p[..., 1:]
    s = qs * ps - torch.sum(qv * pv, dim=-1, keepdim=True)
    v = qs * pv + ps * qv + torch.cross(qv, pv, dim=-1)
    return torch.cat([s, v], dim=-1)

def quaternion_exponential(v):

    norm_v = torch.norm(v, dim=-1, keepdim=True)
    mask = (norm_v.squeeze(-1) < 1e-8)
    result = torch.zeros(v.shape[:-1] + (4,), dtype=v.dtype, device=v.device)

    if mask.any():
        result[mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=v.dtype, device=v.device)
    
    if (~mask).any():
        norm_v_valid = norm_v[~mask]
        v_valid = v[~mask]
        qs = torch.cos(norm_v_valid)
        qv = (v_valid / norm_v_valid) * torch.sin(norm_v_valid)
        result[~mask] = torch.cat([qs, qv], dim=-1)
    
    return result

def quaternion_inverse(q):
    q_conj = torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)
    return q_conj / (torch.linalg.norm(q, dim=-1, keepdim=True) ** 2)

def quaternion_log(q):
    qs, qv = q[..., 0:1], q[..., 1:]
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    qv_norm = torch.clamp(torch.norm(qv, dim=-1, keepdim=True), min=1e-8)
    qs_normalized = qs / torch.clamp(q_norm, min=1e-8)
    
    eps = 1e-7
    theta = torch.acos(torch.clamp(qs_normalized, -1.0 + eps, 1.0 - eps))
    
    log_s = torch.log(torch.clamp(q_norm, min=1e-8))
    log_v = (qv / qv_norm) * theta
    
    return torch.cat([log_s, log_v], dim=-1)

def update_quaternion(qt, tau, wx, wy, wz):
    if qt.dim() == 1:
        qt = qt.unsqueeze(0)
        
    omega = torch.tensor([[wx, wy, wz]], dtype=qt.dtype, device=qt.device)
    
    delta_q = quaternion_exponential(tau * omega / 2.0)
    qt1 = quaternion_multiplication(qt, delta_q)

    qt1 = qt1 / torch.norm(qt1, dim=-1, keepdim=True)
    
    return qt1.squeeze(0)

def cost_function(qt, omega, at, tau):
    T = qt.shape[0] - 1

    # Motion Model error
    q_pred = quaternion_multiplication(qt[:-1], quaternion_exponential((tau[:T].unsqueeze(-1) * omega[:T]) / 2.0))
    motion_error = 2.0 * quaternion_log(quaternion_multiplication(quaternion_inverse(q_pred), qt[1:]))
    motion_cost = 0.5 * torch.sum(motion_error ** 2)
    
    # Observation Model Error
    gravity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=qt.device).unsqueeze(0).expand(qt.shape[0], -1)
    h_qt = quaternion_multiplication(quaternion_inverse(qt), quaternion_multiplication(gravity, qt))
    observation_error = at - h_qt[:, 1:]
    observation_cost = 0.5 * torch.sum(observation_error ** 2)

    return motion_cost + observation_cost