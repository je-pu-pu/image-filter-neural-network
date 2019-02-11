import os
import sys
import math
import glob
import datetime
import numpy as np
from PIL import Image

import tensorflow.keras as keras

# チャンネル数
channels = 3

# 学習データ・入力画像のサイズ
w = 512
h = 512

# ラベルデータ・出力画像のサイズ
ow = 512
oh = 512


def load_image(file_path, resize=True):
    with Image.open(file_path) as im:
        im = im.convert("RGB")

        if resize:
            im = im.resize((w, h), Image.BICUBIC)

        x = np.array(im)
        x = x / 255.0
        # x = x.flatten()

    return x


def save_image(x, file_path):
    x = x.reshape([oh, ow, channels])
    x = np.uint8(x * 255)

    im = Image.fromarray(x)
    im.save(file_path)


class Generator(keras.utils.Sequence):

    def __init__(self, path_list):
        self.path_list = path_list
        self.batch_size = 32

    def __getitem__(self, n):

        n *= self.batch_size

        path_list = self.path_list[n:n+self.batch_size]

        # print( len( path_list ) )
        # print( path_list )
        # return

        x = []
        y = []

        for path in path_list:
            file_name = os.path.basename(path)

            x.append(load_image("./x/" + file_name, True))
            y.append( load_image( "./x/" + file_name, False ) )
            # y.append(load_image("./y/" + file_name, False))

        v = (np.array(x, dtype=np.float32), np.array(y, dtype=np.float32))

        # print( 'x', v[ 0 ].shape )
        # print( 'y', v[ 1 ].shape )

        return v

    def __len__(self):
        return int(math.ceil(len(self.path_list) / self.batch_size))

    def on_epoch_end(self):
        pass


class SaveImageCallback(keras.callbacks.Callback):
    def __init__(self):
        self.images = [
            load_image("./src.png", True),
            load_image("./src2.png", True),
        ]

    def on_epoch_end(self, epoch, logs={}):
        for n, im in enumerate(self.images):
            y = model.predict(np.array([im]))
            save_image(y[0], "./output/" + str(n) + "-" + str(epoch) + ".png")


def build_model():
    model = keras.models.Sequential()
    model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 5, 5 ), padding="same" ) )
    model.add( keras.layers.LeakyReLU() )
    model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 5, 5 ), padding="same" ) )
    model.add( keras.layers.LeakyReLU() )
    model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 5, 5 ), padding="same" ) )
    model.add( keras.layers.LeakyReLU() )
    model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 5, 5 ), padding="same" ) )
    model.add( keras.layers.LeakyReLU() )
    model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 5, 5 ), padding="same" ) )
    model.add( keras.layers.LeakyReLU() )
    
    # model.add( keras.layers.Conv2D( 16, kernel_size=( 17, 17 ), padding="same", activation="relu" ) )
    # model.add( keras.layers.Conv2D( 3, kernel_size=( 17, 17 ), padding="same", activation="relu" ) )

    # model.add( keras.layers.Reshape( ( oh, ow, channels ) ) )
    # model.add( keras.layers.Dense( w * h * channels, activation="relu" ) )

    # model.add( keras.layers.Flatten() )

    # model.add( keras.layers.Reshape( ( oh, ow, channels ) ) )

    # model.add( keras.layers.Activation( "sigmoid" ) )
    # model.add( keras.layers.Dense( 100, activation="sigmoid" ) )

    # model.add( keras.layers.Conv2D( 3, kernel_size=( 2, 2 ), padding="same", activation="relu" ) )

    # model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 2, 2 ), activation="sigmoid" ) )
    # model.add( keras.layers.AveragePooling2D( pool_size=( 8, 8 ) ) )
    # model.add(keras.layers.Flatten(input_shape=(h, w, channels)))
    # model.add( keras.layers.Dense( 8 * 8, activation="sigmoid" ) )
    # model.add(keras.layers.Dense(oh * ow * channels, activation="sigmoid"))
    # model.add(keras.layers.Reshape((oh, ow, channels)))

    model.compile(loss='mean_squared_error', optimizer='adam')  # 回帰問題

    return model


def load_model(model_file_path):
    model = keras.models.load_model(model_file_path)
    print("model file loaded. : " + model_file_path)

    return model


if len(sys.argv) >= 2:
    model = load_model(sys.argv[1])
else:
    model = build_model()

dt = datetime.datetime.now().strftime("%Y%m%d-%H%M")

save_image_cb = SaveImageCallback()
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, verbose=1)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "./models/" + dt + "-{epoch:06d}-{loss:.4f}-{val_loss:.4f}.hdf5", monitor="val_loss", period=10)
# tf_log = keras.callbacks.TensorBoard( log_dir='./logs/log/', histogram_freq=0 )

path_list = glob.glob("./x/*.png")

gen_train = Generator(path_list[:-50])
gen_val = Generator(path_list[-50:])

model.fit_generator(gen_train, epochs=1000, callbacks=[
                    save_image_cb, model_checkpoint], validation_data=gen_val)
