
import tensorflow as tf



# ライブラリ「TensorFlow」のtensorflowパッケージを「tf」という別名でインポート
import tensorflow as tf
from tensorflow.keras import layers       # 「レイヤーズ」モジュールのインポート

# 定数（モデル定義時に必要となる数値）
INPUT_FEATURES = 2  # 入力（特徴）の数： 2
LAYER1_NEURONS = 3  # ニューロンの数： 3
LAYER2_NEURONS = 3  # ニューロンの数： 3
OUTPUT_RESULTS = 1  # 出力結果の数： 1



# ### 活性化関数を変数（ハイパーパラメーター）として定義 ###
# 変数（モデル定義時に必要となる数値）
activation1 = layers.Activation('tanh' # 活性化関数（隠れ層用）： tanh関数（変更可能）
    , name='activation1'               # 活性化関数にも名前付け
    )
activation2 = layers.Activation('tanh' # 活性化関数（隠れ層用）： tanh関数（変更可能）
    , name='activation2'               # 活性化関数にも名前付け
    )
acti_out = layers.Activation('tanh'    # 活性化関数（出力層用）： tanh関数（固定）
    , name='acti_out'                  # 活性化関数にも名前付け
    )

# tf.keras.Modelをサブクラス化してモデルを定義
class NeuralNetwork(tf.keras.Model):

    # ### レイヤーを定義 ###
    def __init__(self, *args, **kwargs):
        super(NeuralNetwork, self).__init__(*args, **kwargs)

        # 入力層は定義「不要」。実際の入力によって決まるので

        # 隠れ層：1つ目のレイヤー
        self.layer1 = layers.Dense(          # 全結合層
            #input_shape=(INPUT_FEATURES,),  # 入力層（定義不要）
            name='layer1',                   # 表示用に名前付け
            units=LAYER1_NEURONS)            # ユニットの数

        # 隠れ層：2つ目のレイヤー
        self.layer2 = layers.Dense(          # 全結合層
            name='layer2',                   # 表示用に名前付け
            units=LAYER2_NEURONS)            # ユニットの数

        # 出力層
        self.layer_out = layers.Dense(       # 全結合層
            name='layer_out',                # 表示用に名前付け
            units=OUTPUT_RESULTS)            # ユニットの数

    # ### フィードフォワードを定義 ###
    def call(self, inputs, training=None): # 入力と、訓練／評価モード
        # 「出力＝活性化関数（第n層（入力））」の形式で記述
        x1 = activation1(self.layer1(inputs))     # 活性化関数は変数として定義
        x2 = activation2(self.layer2(x1))         # 同上
        outputs = acti_out(self.layer_out(x2))    # ※活性化関数は「tanh」固定
        return outputs

# ### 以上でモデル設計は完了 ###
#model.summary()  # モデル内容の出力はできない！（後述）

model1 = NeuralNetwork(name='subclassing_model1') # モデルの生成

# 方法1： 推論して動的にグラフを構築する
temp_input = [[0.1,-0.2]]                         # 仮の入力値
temp_output = model1.predict(temp_input)          # 推論の実行

# モデルの内容を出力
model1.summary()