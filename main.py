import os
import sys
import math
import glob
import datetime
import numpy as np
from PIL import Image

import tensorflow.keras as keras

from argparse import ArgumentParser

# チャンネル数
channels = 3

# 学習データ・入力画像のサイズ
w = 512
h = 512

# ラベルデータ・出力画像のサイズ
ow = 512
oh = 512

# 画像を読み込む
def load_image(file_path, resize=True):
    with Image.open(file_path) as im:
        im = im.convert("RGB")

        if resize:
            im = im.resize((w, h), Image.BICUBIC)

        x = np.array(im)
        x = x / 255.0
        # x = x.flatten()

    return x

# 画像を保存する
def save_image( x, file_path, w=ow, h=oh ):
    x = x.reshape( [ h, w, channels ] )
    x = np.uint8( x * 255 )

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
            file_name = os.path.basename( path )

            x.append( load_image( "./x/" + file_name, True ) )
            y.append( load_image( "./y/" + file_name, False ) )

        v = ( np.array( x, dtype=np.float32 ), np.array( y, dtype=np.float32 ) )

        # print( 'x', v[ 0 ].shape )
        # print( 'y', v[ 1 ].shape )

        return v

    def __len__(self):
        return int(math.ceil(len(self.path_list) / self.batch_size))

    def on_epoch_end(self):
        pass

# 画像を予測 ( = 変換 ) し、ファイルに保存する
def predict_image( image, output_file_path ):
    y = model.predict( np.array( [ image ] ) )
    save_image( y[ 0 ], output_file_path )

class SaveImageCallback(keras.callbacks.Callback):
    def __init__(self):
        self.images = [
            load_image("./src.png", True),
            load_image("./src2.png", True),
        ]
        
        for n in range( len( self.images ) ):
            os.makedirs( "./output/" + str( n ), exist_ok=True )

    def on_epoch_end( self, epoch, logs={} ):
        for n, im in enumerate( self.images ):
            predict_image( im, "./output/" + str( n ) + "/" + str( n ) + "-" + str( epoch ) + ".png" )


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

    model.compile(loss='mean_squared_error', optimizer='adam')  # 回帰問題

    return model

# 畳み込みモデル
def build_conv_model():
    model = keras.models.Sequential()
    model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 5, 5 ), padding="same" ) )
    model.add( keras.layers.LeakyReLU() )
    
    model.compile(loss='mean_squared_error', optimizer='adam')  # 回帰問題
    
    return model

# 畳み込みモデル ( 輪郭検出用 )
def build_conv_model_for_border():
    model = keras.models.Sequential()
    model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 5, 5 ), padding="same", activation="sigmoid" ) ) # sigmoid を使うことにより各色の値が 0 ～ 1 に上手く収まり絵が破綻しない
    model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 5, 5 ), padding="same", activation="sigmoid" ) )
    model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 5, 5 ), padding="same", activation="sigmoid" ) )
    
    model.compile( loss="mean_squared_error", optimizer="adam" ) # 回帰問題
    
    return model

# 全結合モデル
def build_dense_model():
    model = keras.models.Sequential()
    # model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 2, 2 ), activation="sigmoid" ) )
    # model.add( keras.layers.AveragePooling2D( pool_size=( 8, 8 ) ) )
    # 
    model.add( keras.layers.Flatten( input_shape=( h, w, channels ) ) )
    model.add( keras.layers.Dense( 8 * 8, activation="sigmoid" ) )
    model.add( keras.layers.Dense( oh * ow * channels, activation="sigmoid" ) )
    model.add( keras.layers.Reshape( ( oh, ow, channels ) ) )

    model.compile(loss='mean_squared_error', optimizer='adam')  # 回帰問題

    return model

def load_model(model_file_path):
    model = keras.models.load_model(model_file_path)
    print("model file loaded. : " + model_file_path)

    return model

arg_parser = ArgumentParser()
arg_parser.add_argument( '-e', '--epoch', type=int, default=0, help='initial epoch' ) # 学習途中のモデルを読み込んで学習を再開する時に指定する開始エポック数
arg_parser.add_argument( '-m', '--model', help='file path to model ( .hdf5 )' )
arg_parser.add_argument( '-p', '--predict', help='file path to image to predict' )
arg_parser.add_argument( '-w', '--width', type=int, help='input image width & height' )
arg_parser.add_argument( '-ow', '--output-width', type=int, help='output image width & height' )
arg_parser.add_argument( '-o', '--output-file-path', default='./predicted.png', help='output predicted image file path' )
args = arg_parser.parse_args()

if args.width:
    w = args.width
    h = args.width

if args.output_width:
    ow = args.output_width
    oh = args.output_width

if args.model:
    model = load_model( args.model )
else:
    model = build_conv_model_for_border()
    # model = build_dense_model()

if args.predict:
    keras.utils.plot_model( model, to_file="./model.png", show_shapes=True )
    
    image = load_image( args.predict, True )
    save_image( image, './resized.png', w, h )
    predict_image( image, args.output_file_path )
    print( "predicted." )
    exit()

os.makedirs( "./output/models", exist_ok=True )

dt = datetime.datetime.now().strftime( "%Y%m%d-%H%M" )

keras.utils.plot_model( model, to_file="./output/model.png", show_shapes=True )

save_image_cb = SaveImageCallback()
early_stopping = keras.callbacks.EarlyStopping( monitor='val_loss', patience=10, verbose=1 )
model_checkpoint = keras.callbacks.ModelCheckpoint( "./output/models/" + dt + "-{epoch:06d}-{loss:.4f}-{val_loss:.4f}.hdf5", monitor="val_loss", period=20, save_best_only=True )
# tf_log = keras.callbacks.TensorBoard( log_dir='./logs/log/', histogram_freq=0 )

path_list = glob.glob( "./x/*.png" )

# バリデーションに使う画像の数
validation_data_count = 50

training_path_list = path_list[:-validation_data_count]
validation_path_list = path_list[-validation_data_count:]

gen_train = Generator( training_path_list )
gen_val = Generator( validation_path_list )

model.fit_generator( gen_train, initial_epoch=args.epoch, epochs=args.epoch + 1000, callbacks=[ save_image_cb, model_checkpoint ], validation_data=gen_val )
