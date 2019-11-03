# Encoder
x = Input(shape=(max_length, n_vocab))
h = LSTM(lstm_dim, return_sequences=False, name='lstm_1')(x)
z_mean = Dense(latent_dim)(h) # 潜在変数の平均 μ
z_log_var = Dense(latent_dim)(h) #潜在変数の分散 σのlog
encoder = Model(inputs=x, outputs=[z_mean, z_log_var])

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon
    
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# Decoder
decoder_input = Input(shape=(latent_dim,))
repeated_context = RepeatVector(max_length)(decoder_input)
h_decoded = LSTM(lstm_dim, return_sequences=True, name='dec_lstm_1')(repeated_context)
decoder_output = TimeDistributed(Dense(n_vocab, activation='softmax'), name='decoded_mean')(h_decoded)
decoder = Model(inputs=decoder_input, outputs=decoder_output)

x_decoded = decoder(z)
