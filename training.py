import os
import numpy as np
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import weights_init
import matplotlib.pyplot as plt
from data_loader import MNIST_Paired
from torch.utils.data import DataLoader
from networks import Encoder, Decoder, Discriminator
from utils import imshow_grid, mse_loss, reparameterize, transform_config


def training_procedure(FLAGS):
    """
    model definition
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    encoder.apply(weights_init)

    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder.apply(weights_init)

    discriminator = Discriminator()
    discriminator.apply(weights_init)

    # load saved models if load_saved flag is true
    if FLAGS.load_saved:
        encoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.encoder_save)))
        decoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.decoder_save)))
        discriminator.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.discriminator_save)))

    """
    variable definition
    """
    real_domain_labels = 1
    fake_domain_labels = 0

    X_1 = torch.FloatTensor(FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_size, FLAGS.image_size)
    X_2 = torch.FloatTensor(FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_size, FLAGS.image_size)
    X_3 = torch.FloatTensor(FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_size, FLAGS.image_size)

    domain_labels = torch.LongTensor(FLAGS.batch_size)
    style_latent_space = torch.FloatTensor(FLAGS.batch_size, FLAGS.style_dim)

    """
    loss definitions
    """
    cross_entropy_loss = nn.CrossEntropyLoss()

    '''
    add option to run on GPU
    '''
    if FLAGS.cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()

        cross_entropy_loss.cuda()

        X_1 = X_1.cuda()
        X_2 = X_2.cuda()
        X_3 = X_3.cuda()

        domain_labels = domain_labels.cuda()
        style_latent_space = style_latent_space.cuda()

    """
    optimizer definition
    """
    auto_encoder_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    discriminator_optimizer = optim.Adam(
        list(discriminator.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    generator_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    """
    training
    """
    if torch.cuda.is_available() and not FLAGS.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # load_saved is false when training is started from 0th iteration
    if not FLAGS.load_saved:
        with open(FLAGS.log_file, 'w') as log:
            log.write('Epoch\tIteration\tReconstruction_loss\tKL_divergence_loss\t')
            log.write('Generator_loss\tDiscriminator_loss\tDiscriminator_accuracy\n')

    # load data set and create data loader instance
    print('Loading MNIST paired dataset...')
    paired_mnist = MNIST_Paired(root='mnist', download=True, train=True, transform=transform_config)
    loader = cycle(DataLoader(paired_mnist, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))

    # initialise variables
    discriminator_accuracy = 0.

    # initialize summary writer
    writer = SummaryWriter()

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print('')
        print('Epoch #' + str(epoch) + '..........................................................................')

        for iteration in range(int(len(paired_mnist) / FLAGS.batch_size)):
            # A. run the auto-encoder reconstruction
            image_batch_1, image_batch_2, _ = next(loader)

            auto_encoder_optimizer.zero_grad()

            X_1.copy_(image_batch_1)
            X_2.copy_(image_batch_2)

            style_mu_1, style_logvar_1, class_1 = encoder(Variable(X_1))
            style_1 = reparameterize(training=True, mu=style_mu_1, logvar=style_logvar_1)

            kl_divergence_loss_1 = - 0.5 * torch.sum(1 + style_logvar_1 - style_mu_1.pow(2) - style_logvar_1.exp())
            kl_divergence_loss_1 /= (FLAGS.batch_size * FLAGS.num_channels * FLAGS.image_size * FLAGS.image_size)
            kl_divergence_loss_1.backward(retain_graph=True)

            _, __, class_2 = encoder(Variable(X_2))

            reconstructed_X_1 = decoder(style_1, class_1)
            reconstructed_X_2 = decoder(style_1, class_2)

            reconstruction_error_1 = mse_loss(reconstructed_X_1, Variable(X_1))
            reconstruction_error_1.backward(retain_graph=True)

            reconstruction_error_2 = mse_loss(reconstructed_X_2, Variable(X_1))
            reconstruction_error_2.backward()

            reconstruction_error = reconstruction_error_1 + reconstruction_error_2
            kl_divergence_error = kl_divergence_loss_1

            auto_encoder_optimizer.step()

            # B. run the generator
            for i in range(FLAGS.generator_times):

                generator_optimizer.zero_grad()

                image_batch_1, _, __ = next(loader)
                image_batch_3, _, __ = next(loader)

                domain_labels.fill_(real_domain_labels)
                X_1.copy_(image_batch_1)
                X_3.copy_(image_batch_3)

                style_mu_1, style_logvar_1, _ = encoder(Variable(X_1))
                style_1 = reparameterize(training=True, mu=style_mu_1, logvar=style_logvar_1)

                kl_divergence_loss_1 = - 0.5 * torch.sum(1 + style_logvar_1 - style_mu_1.pow(2) - style_logvar_1.exp())
                kl_divergence_loss_1 /= (FLAGS.batch_size * FLAGS.num_channels * FLAGS.image_size * FLAGS.image_size)
                kl_divergence_loss_1.backward(retain_graph=True)

                _, __, class_3 = encoder(Variable(X_3))
                reconstructed_X_1_3 = decoder(style_1, class_3)

                output_1 = discriminator(Variable(X_3), reconstructed_X_1_3)

                generator_error_1 = cross_entropy_loss(output_1, Variable(domain_labels))
                generator_error_1.backward(retain_graph=True)

                style_latent_space.normal_(0., 1.)
                reconstructed_X_latent_3 = decoder(Variable(style_latent_space), class_3)

                output_2 = discriminator(Variable(X_3), reconstructed_X_latent_3)

                generator_error_2 = cross_entropy_loss(output_2, Variable(domain_labels))
                generator_error_2.backward()

                generator_error = generator_error_1 + generator_error_2
                kl_divergence_error += kl_divergence_loss_1

                generator_optimizer.step()

            # C. run the discriminator
            for i in range(FLAGS.discriminator_times):

                discriminator_optimizer.zero_grad()

                # train discriminator on real data
                domain_labels.fill_(real_domain_labels)

                image_batch_1, _, __ = next(loader)
                image_batch_2, image_batch_3, _ = next(loader)

                X_1.copy_(image_batch_1)
                X_2.copy_(image_batch_2)
                X_3.copy_(image_batch_3)

                real_output = discriminator(Variable(X_2), Variable(X_3))

                discriminator_real_error = cross_entropy_loss(real_output, Variable(domain_labels))
                discriminator_real_error.backward()

                # train discriminator on fake data
                domain_labels.fill_(fake_domain_labels)

                style_mu_1, style_logvar_1, _ = encoder(Variable(X_1))
                style_1 = reparameterize(training=False, mu=style_mu_1, logvar=style_logvar_1)

                _, __, class_3 = encoder(Variable(X_3))
                reconstructed_X_1_3 = decoder(style_1, class_3)

                fake_output = discriminator(Variable(X_3), reconstructed_X_1_3)

                discriminator_fake_error = cross_entropy_loss(fake_output, Variable(domain_labels))
                discriminator_fake_error.backward()

                # total discriminator error
                discriminator_error = discriminator_real_error + discriminator_fake_error

                # calculate discriminator accuracy for this step
                target_true_labels = torch.cat((torch.ones(FLAGS.batch_size), torch.zeros(FLAGS.batch_size)), dim=0)
                if FLAGS.cuda:
                    target_true_labels = target_true_labels.cuda()

                discriminator_predictions = torch.cat((real_output, fake_output), dim=0)
                _, discriminator_predictions = torch.max(discriminator_predictions, 1)

                discriminator_accuracy = (discriminator_predictions.data == target_true_labels.long()
                                          ).sum().item() / (FLAGS.batch_size * 2)

                if discriminator_accuracy < FLAGS.discriminator_limiting_accuracy:
                    discriminator_optimizer.step()

            if (iteration + 1) % 50 == 0:
                print('')
                print('Epoch #' + str(epoch))
                print('Iteration #' + str(iteration))

                print('')
                print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
                print('KL-Divergence loss: ' + str(kl_divergence_error.data.storage().tolist()[0]))

                print('')
                print('Generator loss: ' + str(generator_error.data.storage().tolist()[0]))
                print('Discriminator loss: ' + str(discriminator_error.data.storage().tolist()[0]))
                print('Discriminator accuracy: ' + str(discriminator_accuracy))

                print('..........')

            # write to log
            with open(FLAGS.log_file, 'a') as log:
                log.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n'.format(
                    epoch,
                    iteration,
                    reconstruction_error.data.storage().tolist()[0],
                    kl_divergence_error.data.storage().tolist()[0],
                    generator_error.data.storage().tolist()[0],
                    discriminator_error.data.storage().tolist()[0],
                    discriminator_accuracy
                ))

            # write to tensorboard
            writer.add_scalar('Reconstruction loss', reconstruction_error.data.storage().tolist()[0],
                              epoch * (int(len(paired_mnist) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('KL-Divergence loss', kl_divergence_error.data.storage().tolist()[0],
                              epoch * (int(len(paired_mnist) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('Generator loss', generator_error.data.storage().tolist()[0],
                              epoch * (int(len(paired_mnist) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('Discriminator loss', discriminator_error.data.storage().tolist()[0],
                              epoch * (int(len(paired_mnist) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('Discriminator accuracy', discriminator_accuracy * 100,
                              epoch * (int(len(paired_mnist) / FLAGS.batch_size) + 1) + iteration)

        # save model after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.end_epoch:
            torch.save(encoder.state_dict(), os.path.join('checkpoints', FLAGS.encoder_save))
            torch.save(decoder.state_dict(), os.path.join('checkpoints', FLAGS.decoder_save))
            torch.save(discriminator.state_dict(), os.path.join('checkpoints', FLAGS.discriminator_save))
