import numpy as np

def quaternion_multiplication(q, p):
    qs, qv = q[0], q[1:]
    ps, pv = p[0], p[1:]
    s = qs * ps - np.dot(qv, pv)
    v = qs * pv + ps * qv + np.cross(qv, pv)
    return np.hstack([[s], v])

def quaternion_exponential(v):
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])

    qs = np.cos(norm_v)
    qv = (v / norm_v) * np.sin(norm_v)
    return np.hstack([[qs], qv])

def quaternion_inverse(q):
    return q / (np.linalg.norm(q) ** 2)

def quaternion_log(q):
    qs, qv = q[0], q[1:]
    q_norm = np.linalg.norm(q)
    qv_norm = np.linalg.norm(q)
    return [np.log(q_norm), qv / qv_norm * np.arccos(qs / q_norm)]

def update_quaternion(qt, tau, wx, wy, wz):
    omega = np.array([wx, wy, wz])
    delta_q = quaternion_exponential(tau * omega / 2)
    qt1 = quaternion_multiplication(qt, delta_q)
    qt1 = qt1 / np.linalg.norm(qt1)
    return qt1