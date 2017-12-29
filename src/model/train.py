import os
import sys
import time
import numpy as np
import models
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
# Utils
sys.path.append("../utils")
import general_utils
import data_utils

# disc_loss_list = list()
disc_n = 1
disc_prev_avg = 0

# gen1_loss_list = list()
gen1_n = 1
gen1_prev_avg = 0

# gen2_loss_list = list()
gen2_n = 1
gen2_prev_avg = 0

# gen3_loss_list = list()
gen3_n = 1
gen3_prev_avg = 0

def running_avg(disc_loss, gen1, gen2, gen3):

    # global disc_loss_list, disc_prev_avg, gen1_loss_list, gen1_prev_avg, gen2_loss_list, gen2_prev_avg, gen2_loss_list, gen2_prev_avg
    global disc_n, disc_prev_avg, gen1_n, gen1_prev_avg, gen2_n, gen2_prev_avg, gen3_n, gen3_prev_avg

    # disc_loss_list.append(disc_loss)
    # loss_list_n = len(disc_loss_list)
    disc_new_avg = ((disc_n-1)*disc_prev_avg + disc_loss)/disc_n
    disc_prev_avg = disc_new_avg
    disc_n += 1

    # gen1_loss_list.append(gen1)
    # loss_list_n = len(gen1_loss_list)
    gen1_new_avg = ((gen1_n-1)*gen1_prev_avg + gen1)/gen1_n
    gen1_prev_avg = gen1_new_avg
    gen1_n += 1

    # gen2_loss_list.append(gen2)
    # loss_list_n = len(gen2_loss_list)
    gen2_new_avg = ((gen2_n-1)*gen2_prev_avg + gen2)/gen2_n
    gen2_prev_avg = gen2_new_avg
    gen2_n += 1

    # gen3_loss_list.append(gen3)
    # loss_list_n = len(gen3_loss_list)
    gen3_new_avg = ((gen3_n-1)*gen3_prev_avg + gen3)/gen3_n
    gen3_prev_avg = gen3_new_avg
    gen3_n += 1

    return disc_new_avg, gen1_new_avg, gen2_new_avg, gen3_new_avg

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    model_name = kwargs["model_name"]
    generator = kwargs["generator"]
    image_dim_ordering = kwargs["image_dim_ordering"]
    img_dim = kwargs["img_dim"]
    patch_size = kwargs["patch_size"]
    bn_mode = kwargs["bn_mode"]
    label_smoothing = kwargs["use_label_smoothing"]
    label_flipping = kwargs["label_flipping"]
    dset = kwargs["dset"]
    use_mbd = kwargs["use_mbd"]

    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name)
    print "hi"

    # Load and rescale data
    X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_dim_ordering)
    img_dim = X_full_train.shape[-3:]
    print "data loaded in memory"

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_dim_ordering)

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
        generator_model = models.load("generator_unet_%s" % generator,
                                      img_dim,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size)
        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator",
                                          img_dim_disc,
                                          nb_patch,
                                          bn_mode,
                                          use_mbd,
                                          batch_size)

        generator_model.compile(loss='mae', optimizer=opt_discriminator)
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   img_dim,
                                   patch_size,
                                   image_dim_ordering)

        loss = [l1_loss, 'binary_crossentropy']
        loss_weights = [1E1, 1]
        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        discriminator_model.trainable = True
        discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        gen_loss = None
        disc_loss = None

        iter_num = 17
        weights_path = "/home/abhik/pix2pix/src/model/weights/gen_weights_iter%s_epoch20.h5" % (str(iter_num - 1))
        print weights_path
        generator_model.load_weights(weights_path)
        
        #discriminator_model.load_weights("disc_weights1.2.h5")

        #DCGAN_model.load_weights("DCGAN_weights1.2.h5")
        print ("Weights Loaded for iter - %d" % iter_num)

        # Running average
        losses_list = list()
        # loss_list = list()
        # prev_avg = 0

        # Start training
        print("Start training")
        for e in range(nb_epoch):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            # global disc_n, disc_prev_avg, gen1_n, gen1_prev_avg, gen2_n, gen2_prev_avg, gen3_n, gen3_prev_avg

            # disc_n = 1
            # disc_prev_avg = 0

            # gen1_n = 1
            # gen1_prev_avg = 0

            # gen2_n = 1
            # gen2_prev_avg = 0

            # gen3_n = 1
            # gen3_prev_avg = 0

            for X_full_batch, X_sketch_batch in data_utils.gen_batch(X_full_train, X_sketch_train, batch_size):

                # Create a batch to feed the discriminator model
                X_disc, y_disc = data_utils.get_disc_batch(X_full_batch,
                                                           X_sketch_batch,
                                                           generator_model,
                                                           batch_counter,
                                                           patch_size,
                                                           image_dim_ordering,
                                                           label_smoothing=label_smoothing,
                                                           label_flipping=label_flipping)

                # Update the discriminator
                disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)

                # Create a batch to feed the generator model
                X_gen_target, X_gen = next(data_utils.gen_batch(X_full_train, X_sketch_train, batch_size))
                y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                y_gen[:, 1] = 1

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                # Running average
                # loss_list.append(disc_loss)
                # loss_list_n = len(loss_list)
                # new_avg = ((loss_list_n-1)*prev_avg + disc_loss)/loss_list_n
                # prev_avg = new_avg

                # disc_avg, gen1_avg, gen2_avg, gen3_avg = running_avg(disc_loss, gen_loss[0], gen_loss[1], gen_loss[2])

                # print("running disc loss", new_avg)
                # print(disc_loss, gen_loss)
                # print ("all losses", disc_avg, gen1_avg, gen2_avg, gen3_avg)
                # print("")

                batch_counter += 1
                progbar.add(batch_size, values=[("D logloss", disc_loss),
                                                ("G tot", gen_loss[0]),
                                                ("G L1", gen_loss[1]),
                                                ("G logloss", gen_loss[2])])

                # Saving data for plotting
                # losses = [e+1, batch_counter, disc_loss, gen_loss[0], gen_loss[1], gen_loss[2], disc_avg, gen1_avg, gen2_avg, gen3_avg, iter_num]
                # losses_list.append(losses)

                # Save images for visualization
                if batch_counter % (n_batch_per_epoch / 2) == 0:
                    # Get new images from validation
                    data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                    batch_size, image_dim_ordering, "training", iter_num)
                    X_full_batch, X_sketch_batch = next(data_utils.gen_batch(X_full_val, X_sketch_val, batch_size))
                    data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                    batch_size, image_dim_ordering, "validation", iter_num)

                if batch_counter >= n_batch_per_epoch:
                    break

            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

            #Running average
            disc_avg, gen1_avg, gen2_avg, gen3_avg = running_avg(disc_loss, gen_loss[0], gen_loss[1], gen_loss[2])

            #Validation loss
            y_gen_val = np.zeros((X_sketch_batch.shape[0], 2), dtype=np.uint8)
            y_gen_val[:, 1] = 1
            val_loss = DCGAN_model.test_on_batch(X_full_batch, [X_sketch_batch, y_gen_val])
            # print "val_loss ===" + str(val_loss)

            #logging
            # Saving data for plotting
            losses = [e+1, iter_num, disc_loss, gen_loss[0], gen_loss[1], gen_loss[2], disc_avg, gen1_avg, gen2_avg, gen3_avg, val_loss[0], val_loss[1], val_loss[2]]
            losses_list.append(losses)

            if (e+1) % 5 == 0:
                gen_weights_path = os.path.join('../../models/%s/gen_weights_iter%s_epoch%s.h5' % (model_name, iter_num, e+1))
                generator_model.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join('../../models/%s/disc_weights_iter%s_epoch%s.h5' % (model_name, iter_num, e+1))
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join('../../models/%s/DCGAN_weights_iter%s_epoch%s.h5' % (model_name, iter_num, e+1))
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

                loss_array = np.asarray(losses_list)
                print (loss_array.shape) # 10 element vector

                loss_path = os.path.join('../../losses/loss_iter%s_epoch%s.csv' % (iter_num, e+1))
                np.savetxt(loss_path, loss_array, fmt='%.5f', delimiter=',')
                np.savetxt('test.csv', loss_array, fmt='%.5f', delimiter=',')

    except KeyboardInterrupt:
        pass
