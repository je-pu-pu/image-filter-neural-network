from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Activation, Dropout, BatchNormalization, LeakyReLU
import tensorflow.keras as keras 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


import glob
import os
import math
import numpy as np
from PIL import Image

# チャンネル数
channels = 3

# 学習データ・入力画像のサイズ
w = 512
h = 512

# ラベルデータ・出力画像のサイズ
ow = 512
oh = 512

def load_image( file_path, resize=True ):
    with Image.open( file_path ) as im:
        im = im.convert( "RGB" )

        if resize:
            im = im.resize( ( w, h ), Image.BICUBIC )
        
        x = np.array( im )
        x = x / 255.0
        # x = x.flatten()
    
    return x

def save_image( x, file_path ):
    x = x.reshape( [ oh, ow, channels ] )
    x = np.uint8( x * 255 )
    
    im = Image.fromarray( x )
    im.save( file_path )

class Generator( keras.utils.Sequence ):
    
    def __init__( self, path_list ):
        self.path_list = path_list
        self.batch_size = 32
    
    def __getitem__( self, n ):
        
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
            y.append( load_image( "./x/" + file_name, False ) )
            # y.append( load_image( "./y/" + file_name ) )
            
        v = ( np.array( x, dtype=np.float32 ), np.array( y, dtype=np.float32 ) )
        # v[ 1 ].fill( 1 )

        # print( 'x', v[ 0 ].shape )
        # print( 'y', v[ 1 ].shape )
        
        return v
    
    def __len__( self ):
        return int( math.ceil( len( self.path_list ) / self.batch_size ) )
    
    def on_epoch_end( self ):
        pass

class SaveImageCallback( keras.callbacks.Callback ):
    def on_epoch_end( self, epoch, logs={} ):
        y = model.predict( np.array( [ im ] ) )
        save_image( y[ 0 ], "./output/" + str( epoch ) + ".png" )

im = load_image( "./src.png", True )

path_list = glob.glob( "./x/*.png" )

path_list_train = path_list[:-50]
path_list_val = path_list[-50:]

model = Sequential()
model.add( keras.layers.Conv2D( 3, input_shape=( h, w, channels ), kernel_size=( 9, 9 ), padding="same", activation="relu" ) )
# model.add( keras.layers.Flatten() )
# model.add( keras.layers.Reshape( ( oh, ow, channels ) ) )
# model.add( Dense( w * h * channels, activation="sigmoid" ) )
# model.add( Activation( "sigmoid" ) )
# model.add( Dense( 100, activation="sigmoid" ) )
model.compile( loss='mean_squared_error', optimizer='adam' ) # 回帰問題

save_image_cb = SaveImageCallback()
early_stopping = EarlyStopping( monitor='val_loss', patience=10, verbose=1 )
model_checkpoint = ModelCheckpoint( "./models/10000-{epoch:06d}-{loss:.4f}-{val_loss:.4f}.hdf5", monitor="val_loss", period=10 )
# tf_log = TensorBoard( log_dir='./logs/log/', histogram_freq=0 )

gen_train = Generator( path_list_train )
gen_val = Generator( path_list_val )

model.fit_generator( gen_train, epochs=1000, callbacks=[ save_image_cb, model_checkpoint ], verbose=2, validation_data=gen_val )