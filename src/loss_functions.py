import torch.nn.functional as F
import torch


def sad_loss(outputs, A_pred, S_pred, targets, inputs_A, inputs_S):
    n = outputs[0].shape[1] * outputs[0].shape[2]
    loss = 0
    for output, target in zip(outputs, targets):
        output_flatten = output.reshape(output.shape[0], n)
        output_flatten = F.normalize(output_flatten, dim=0)  # чтобы норма вектора была равна 1

        target_flatten = target.reshape(target.shape[0], n)
        target_flatten = F.normalize(target_flatten, dim=0)

        dot = (output_flatten * target_flatten).sum(axis=0)
        loss += (1 - dot.mean())

    return loss


def sv_loss(X_pred_batch, A_pred, S_pred_batch, sv_X_batch, sv_A_batch, sv_S_batch):
    n = S_pred_batch[0].shape[1] * S_pred_batch[0].shape[2]  # N_COL * N_ROW
    total_loss = 0

    A_pred_normalized = F.normalize(A_pred, dim=1)

    for S_pred, sv_A, sv_S in zip(S_pred_batch, sv_A_batch, sv_S_batch):
        S_pred_flatten = S_pred.reshape(S_pred.shape[0], n)
        S_pred_flatten = F.normalize(S_pred_flatten, dim=0)

        sv_S_flatten = sv_S.reshape(sv_S.shape[0], n)
        sv_S_flatten = F.normalize(sv_S_flatten, dim=0)

        S_dot = (S_pred_flatten * sv_S_flatten).sum(axis=0)
        loss_S = 1 - S_dot.mean()

        sv_A_normalized = F.normalize(sv_A, dim=1)
        A_dot = (A_pred_normalized * sv_A_normalized).sum(axis=1)
        loss_A = 1 - A_dot.mean()

        total_loss += 0.5 * (loss_A + loss_S)

    return total_loss


def balance_loss(X_pred_batch, A_pred, S_pred_batch, sv_X_batch, sv_A_batch, sv_S_batch, alpha=0):
    _sv_loss = sv_loss(X_pred_batch, A_pred, S_pred_batch, sv_X_batch, sv_A_batch, sv_S_batch)
    _sad_loss = sad_loss(X_pred_batch, A_pred, S_pred_batch, sv_X_batch, sv_A_batch, sv_S_batch)
    return alpha * _sad_loss + (1 - alpha) * _sv_loss
