# Generator
generator_input = Input(shape=(max_length, latent_dim,))
x = LSTM(lstm_dim, return_sequences=True)(generator_input)
generator_output = TimeDistributed(Dense(n_vocab, activation='softmax'))(x)
generator = Model(generator_input, generator_output)

# discriminator
discriminator_input = Input(shape=(max_length, n_vocab))
x = LSTM(lstm_dim)(discriminator_input)
dense_output = Dense(256, activation='relu')(x)
discriminator_output = Dense(2, activation='softmax')(dense_output)
discriminator = Model(discriminator_input, discriminator_output)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.1))

# GAN
gan_input = Input(shape=(max_length, latent_dim))
x = generator(gan_input)
gan_output = discriminator(x)
model = Model(gan_input, gan_output)
model.compile(loss='binary_crossentropy', optimizer=opt)
