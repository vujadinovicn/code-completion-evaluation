import torch

def get_disc_loss(real_content, fake_content, disc_content, adversarial_criterion):
    disc_fake_content_hat = disc_content(fake_content.detach())
    disc_fake_content_loss = adversarial_criterion(disc_fake_content_hat, torch.zeros_like(disc_fake_content_hat))
    disc_real_content_hat = disc_content(real_content)
    disc_real_content_loss = adversarial_criterion(disc_real_content_hat, torch.ones_like(disc_real_content_hat))
    return (disc_fake_content_loss + disc_real_content_loss) / 2