#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 19:26:21 2023

@author: luguo
"""


if __name__ == '__main__':

    import argparse
    import itertools
    from torch.utils.data import DataLoader
    from models import *
    from datasets import *
    from utils import *
    import torch
    import sys

    parser = argparse.ArgumentParser(description="pcGAN")
  
    parser.add_argument("-network_name", type=str, default="pcGAN", help="name of the network")
    parser.add_argument("--training_dataset", type=str, default="ex-vivo", help="name of the dataset")
    parser.add_argument("--testing_dataset", type=str, default="ex-vivo", help="name of the testing dataset")
    
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=51, help="number of epochs oef training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=25, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=200, help="size of image height")
    parser.add_argument("--img_width", type=int, default=200, help="size of image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
    
    parser.add_argument("--lambda_merging", type=float, default=10, help="scaling factor for the new loss")
    parser.add_argument("--textfile_training_results_interval", type=int, default=50, help="textfile_training_results_interval")
    
    parser.add_argument("--lambda_cyc", type=float, default=1, help="cycle loss weight")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    
    opt = parser.parse_args()
    print(opt)

    # Create sample and checkpoint directories
    os.makedirs("saved_models/%s-%s" % (opt.network_name, opt.training_dataset), exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    
    criterion_cycle = torch.nn.MSELoss()
    criterion_identity = torch.nn.L1Loss()
    
    #Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    #Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height//2**4, opt.img_width//2**4)
    
    cuda = torch.cuda.is_available()
    input_shape = (opt.channels, opt.img_height, opt.img_width)
    # Initialize generator and discriminator
    G_AB = GeneratorUNet()
    G_CB = GeneratorUNet()
    G_AC = GeneratorUNet()
    
    G_DB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BD = GeneratorResNet(input_shape, opt.n_residual_blocks)
    
    D_AB = Discriminator()
    D_CB = Discriminator()
    D_AC = Discriminator()
    
    D_DB = Discriminator_cycle(input_shape)
    D_BD = discriminator_cycle(input_shape)
    
    
    if cuda:
        G_AB = G_AB.cuda()
        G_CB = G_CB.cuda()
        G_AC = G_AC.cuda()
        
        G_DB = G_DB.cuda()
        G_BD = G_BD.cuda()
        
        D_AB = D_AB.cuda()
        D_CB = D_CB.cuda()
        D_AC = D_AC.cuda()

        D_DB = D_DB.cuda()
        D_BD = D_BD.cuda()
        
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
        
        criterion_cycle.cuda()
        criterion_identity.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("saved_models/%s-%s/G_AB_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))       
        G_CB.load_state_dict(torch.load("saved_models/%s-%s/G_CB_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        G_AC.load_state_dict(torch.load("saved_models/%s-%s/G_AC_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        
        G_DB.load_state_dict(torch.load("saved_models/%s-%s/G_DB_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        G_BD.load_state_dict(torch.load("saved_models/%s-%s/G_BD_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        
        D_AB.load_state_dict(torch.load("saved_models/%s-%s/D_AB_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        D_CB.load_state_dict(torch.load("saved_models/%s-%s/D_CB_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        D_AC.load_state_dict(torch.load("saved_models/%s-%s/D_AC_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        
        D_DB.load_state_dict(torch.load("saved_models/%s-%s/D_DB_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        D_BD.load_state_dict(torch.load("saved_models/%s-%s/D_BD_%d.pth" % (opt.network_name, opt.training_dataset, opt.epoch)))
        
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_CB.apply(weights_init_normal)
        G_AC.apply(weights_init_normal)
        
        G_DB.apply(weights_init_normal)
        G_BD.apply(weights_init_normal)

        D_AB.apply(weights_init_normal)
        D_CB.apply(weights_init_normal)
        D_AC.apply(weights_init_normal)
        
        D_DB.apply(weights_init_normal)
        D_BD.apply(weights_init_normal)
        
    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_CB.parameters(), G_AC.parameters(), G_DB.parameters(), G_BD.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_AB = torch.optim.Adam(D_AB.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_CB = torch.optim.Adam(D_CB.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_AC = torch.optim.Adam(D_AC.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
    optimizer_D_DB = torch.optim.Adam(D_DB.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_BD = torch.optim.Adam(D_BD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_AB = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_AB, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_CB = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_CB, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_AC = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_AC, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    
    lr_scheduler_D_DB = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_DB, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_BD = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_BD, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_AB_buffer = ReplayBuffer()
    fake_CB_buffer = ReplayBuffer()
    fake_AC_buffer = ReplayBuffer()
    
    fake_DB_buffer = ReplayBuffer()
    fake_BD_buffer = ReplayBuffer()

    # Image transformations
    transforms_ = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]


    # Training data loader
    dataloader = DataLoader(
        ImageDataset("../data/Training/%s-training" % opt.training_dataset, transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            real_C = Variable(batch["C"].type(Tensor))
            
            real_D = Variable(batch["D"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()        
            G_CB.train()    
            G_AC.train()
            
            G_DB.train()
            G_BD.train()

            optimizer_G.zero_grad()


            # GAN loss
            fake_AB = G_AB(real_A)
            pred_fake_AB = D_AB(fake_AB,real_A)
            loss_GAN_AB = criterion_GAN(pred_fake_AB, valid)
            
            fake_CB = G_CB(real_C)
            pred_fake_CB = D_CB(fake_CB,real_C)
            loss_GAN_CB = criterion_GAN(pred_fake_CB, valid) 

            fake_AC = G_CB(real_A)
            pred_fake_AC = D_AC(fake_AC,real_A)
            loss_GAN_AC = criterion_GAN(pred_fake_AC, valid)
            
            fake_DB = G_DB(real_D)
            loss_GAN_DB = criterion_GAN(D_DB(fake_DB), valid)
            
            fake_BD = G_BD(real_B)
            loss_GAN_BD = criterion_GAN(D_BD(fake_BD), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_CB + loss_GAN_AC + loss_GAN_DB + loss_GAN_BD) / 5



            # pixel-wise loss
            loss_pixel_AB = criterion_pixelwise(fake_AB,real_B)
            loss_pixel_CB = criterion_pixelwise(fake_CB,real_B)
            loss_pixel_AC = criterion_pixelwise(fake_AC,real_C)
            
            loss_pixel = (loss_pixel_AB + loss_pixel_CB + loss_pixel_AC) / 3
            
            # Cycle loss
            recov_BD = G_BD(fake_DB)
            loss_cycle_BD = criterion_cycle(recov_BD, real_D)
            
            recov_DB = G_DB(fake_BD)
            loss_cycle_DB = criterion_cycle(recov_DB, real_B)
            
            # merging loss:
            recov_2 = G_CB(G_AC(real_A))
            recov_1 = G_AB(real_A)
            loss_merging = criterion_cycle(recov_1, recov_2)
            
            loss_cycle = (loss_cycle_BD + loss_cycle_DB) / 2 + opt.lambda_merging * loss_merging
            
            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel + loss_cycle
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator AB
            # -----------------------

            optimizer_D_AB.zero_grad()

            # Real loss
            pred_real_ABt= D_AB(real_B,real_A)
            loss_real = criterion_GAN(pred_real_ABt,valid)
            # Fake loss (on batch of previously generated samples)
            pred_fake_ABt = D_AB(fake_AB.detach(),real_A)
            loss_fake = criterion_GAN(pred_fake_ABt,fake)
        
            # Total loss
            loss_D_AB = (loss_real + loss_fake) / 2

            loss_D_AB.backward()
            optimizer_D_AB.step()

            # -----------------------
            #  Train Discriminator CB
            # -----------------------

            optimizer_D_CB.zero_grad()

            # Real loss
            pred_real_CBt= D_CB(real_B,real_C)
            loss_real = criterion_GAN(pred_real_CBt,valid)
            # Fake loss (on batch of previously generated samples)
            pred_fake_CBt = D_CB(fake_CB.detach(),real_C)
            loss_fake = criterion_GAN(pred_fake_CBt,fake)
        
            # Total loss
            loss_D_CB = (loss_real + loss_fake) / 2

            loss_D_CB.backward()
            optimizer_D_CB.step()

            # -----------------------
            #  Train Discriminator AC
            # -----------------------

            optimizer_D_AC.zero_grad()

            # Real loss
            pred_real_ACt= D_AC(real_C,real_A)
            loss_real = criterion_GAN(pred_real_ACt,valid)
            # Fake loss (on batch of previously generated samples)
            pred_fake_ACt = D_AC(fake_AC.detach(),real_A)
            loss_fake = criterion_GAN(pred_fake_ACt,fake)
        
            # Total loss
            loss_D_AC = (loss_real + loss_fake) / 2

            loss_D_AC.backward()
            optimizer_D_AC.step()
            
            # -----------------------
            #  Train Discriminator DB
            # -----------------------

            optimizer_D_DB.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_DB(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_DB_ = fake_DB_buffer.push_and_pop(fake_DB)
            loss_fake = criterion_GAN(D_DB(fake_DB_.detach()), fake)
            # Total loss
            loss_D_DB = (loss_real + loss_fake) / 2

            loss_D_DB.backward()
            optimizer_D_DB.step()


            # -----------------------
            #  Train Discriminator BD
            # -----------------------

            optimizer_D_BD.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_BD(real_D), valid)
            # Fake loss (on batch of previously generated samples)
            fake_BD_ = fake_BD_buffer.push_and_pop(fake_BD)
            loss_fake = criterion_GAN(D_BD(fake_BD_.detach()), fake)
            # Total loss
            loss_D_BD = (loss_real + loss_fake) / 2

            loss_D_BD.backward()
            optimizer_D_BD.step()

            loss_D = (loss_D_AB + loss_D_CB + loss_D_AC + loss_D_DB + loss_D_BD) / 5

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),  
                    #loss_identity.item(),
                    time_left,
                )
            )



        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_AB.step()
        lr_scheduler_D_CB.step()
        lr_scheduler_D_AC.step()
        
        lr_scheduler_D_DB.step()
        lr_scheduler_D_BD.step()



        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s-%s/%s-%s-G_AB-%dep.pth" % (opt.network_name,opt.training_dataset, opt.network_name,opt.training_dataset, epoch))
            
            torch.save(G_CB.state_dict(), "saved_models/%s-%s/%s-%s-G_CB-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
           
            torch.save(G_AC.state_dict(), "saved_models/%s-%s/%s-%s-G_AC-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
            
            torch.save(G_DB.state_dict(), "saved_models/%s-%s/%s-%s-G_DB-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
            torch.save(G_BD.state_dict(), "saved_models/%s-%s/%s-%s-G_BD-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
            
            torch.save(D_AB.state_dict(), "saved_models/%s-%s/%s-%s-D_AB-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name,opt.training_dataset, epoch))
            torch.save(D_CB.state_dict(), "saved_models/%s-%s/%s-%s-D_CB-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name,opt.training_dataset, epoch))
            torch.save(D_AC.state_dict(), "saved_models/%s-%s/%s-%s-D_AC-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
           
            torch.save(D_DB.state_dict(), "saved_models/%s-%s/%s-%s-D_DB-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
            torch.save(D_BD.state_dict(), "saved_models/%s-%s/%s-%s-D_BD-%dep.pth" % (opt.network_name, opt.training_dataset, opt.network_name, opt.training_dataset, epoch))
         
