{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled9.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3LAGx-Lqf0Ec",
        "outputId": "73b85607-e110-4045-be89-7601f8fb8289"
      },
      "source": [
        "!pip install keras-tuner"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Collecting keras-tuner\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/20/ec/1ef246787174b1e2bb591c95f29d3c1310070cad877824f907faba3dade9/keras-tuner-1.0.2.tar.gz (62kB)\n",
            "\r\u001b[K     |█████▏                          | 10kB 14.5MB/s eta 0:00:01\r\u001b[K     |██████████▍                     | 20kB 20.1MB/s eta 0:00:01\r\u001b[K     |███████████████▋                | 30kB 10.9MB/s eta 0:00:01\r\u001b[K     |████████████████████▉           | 40kB 8.6MB/s eta 0:00:01\r\u001b[K     |██████████████████████████      | 51kB 5.1MB/s eta 0:00:01\r\u001b[K     |███████████████████████████████▎| 61kB 5.9MB/s eta 0:00:01\r\u001b[K     |████████████████████████████████| 71kB 3.8MB/s \n",
            "\u001b[?25hRequirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (20.9)\n",
            "Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (0.16.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (1.19.5)\n",
            "Requirement already satisfied: tabulate in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (0.8.9)\n",
            "Collecting terminaltables\n",
            "  Downloading https://files.pythonhosted.org/packages/9b/c4/4a21174f32f8a7e1104798c445dacdc1d4df86f2f26722767034e4de4bff/terminaltables-3.1.0.tar.gz\n",
            "Collecting colorama\n",
            "  Downloading https://files.pythonhosted.org/packages/44/98/5b86278fbbf250d239ae0ecb724f8572af1c91f4a11edf4d36a206189440/colorama-0.4.4-py2.py3-none-any.whl\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (4.41.1)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (2.23.0)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (1.4.1)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (0.22.2.post1)\n",
            "Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->keras-tuner) (2.4.7)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (3.0.4)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (1.24.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (2020.12.5)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (2.10)\n",
            "Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->keras-tuner) (1.0.1)\n",
            "Building wheels for collected packages: keras-tuner, terminaltables\n",
            "  Building wheel for keras-tuner (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for keras-tuner: filename=keras_tuner-1.0.2-cp37-none-any.whl size=78938 sha256=f2e157f4012bd0b6312da99d8bae610c0c38ceb6b0ab5a2f3a3a85e0966d8d61\n",
            "  Stored in directory: /root/.cache/pip/wheels/bb/a1/8a/7c3de0efb3707a1701b36ebbfdbc4e67aedf6d4943a1f463d6\n",
            "  Building wheel for terminaltables (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for terminaltables: filename=terminaltables-3.1.0-cp37-none-any.whl size=15356 sha256=d3305383ecb6f210c251d03f6d551c39d234b67ec742aea36453900979574ff9\n",
            "  Stored in directory: /root/.cache/pip/wheels/30/6b/50/6c75775b681fb36cdfac7f19799888ef9d8813aff9e379663e\n",
            "Successfully built keras-tuner terminaltables\n",
            "Installing collected packages: terminaltables, colorama, keras-tuner\n",
            "Successfully installed colorama-0.4.4 keras-tuner-1.0.2 terminaltables-3.1.0\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HNK7zmSwgJ0-"
      },
      "source": [
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "import numpy as np"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "A8Au5OOEggdI"
      },
      "source": [
        "fashion_mnist=keras.datasets.fashion_mnist"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bKzxYWnagtRg",
        "outputId": "11f4ee04-7446-40c7-af50-2d75ee51f59b"
      },
      "source": [
        "(train_images,train_laberls),(test_images,test_labels)=fashion_mnist.load_data()"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz\n",
            "32768/29515 [=================================] - 0s 0us/step\n",
            "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz\n",
            "26427392/26421880 [==============================] - 0s 0us/step\n",
            "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz\n",
            "8192/5148 [===============================================] - 0s 0us/step\n",
            "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz\n",
            "4423680/4422102 [==============================] - 0s 0us/step\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zfTOtdlNhDJl"
      },
      "source": [
        "train_images=train_images/255.0\n",
        "test_images=test_images/255.0"
      ],
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "naTB96erhVba",
        "outputId": "63a020e4-1b1f-40d6-c16b-513f361f1cd5"
      },
      "source": [
        "train_images[0].shape"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(28, 28)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jsCnEx5Chb5G"
      },
      "source": [
        "train_images = train_images.reshape(len(train_images),28,28,1)\n",
        "test_images = test_images.reshape(len(test_images),28,28,1)"
      ],
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8a6ySsa_iFKT"
      },
      "source": [
        "def build_model(hp):\n",
        "  model = keras.Sequential([\n",
        "    keras.layers.Conv2D(\n",
        "        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),\n",
        "        kernel_size=hp.choice('conv_1_kernel', values = [3,5]),\n",
        "        activation='relu',\n",
        "        input_shape=(28,28,1)\n",
        "    ),\n",
        "    keras.layers.Conv2D(\n",
        "        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),\n",
        "        kernel_size=hp.choice('conv_2_kernel', values = [3,5]),\n",
        "        activation='relu'\n",
        "    ),\n",
        "    keras.layers.Flatten(),\n",
        "    keras.layers.Dense(\n",
        "        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),\n",
        "        activation='relu'\n",
        "    ),\n",
        "    keras.layers.Dense(10, activation='softmax')#output layer                        \n",
        "  ])\n",
        "\n",
        "  model.compile(optimizer=keras.optimizers.Adam(hp.choice('learning_rate', values=[1e-2, 1e-3])),\n",
        "              loss='sparse_categorical_crossentropy',\n",
        "              metrics=['accuracy'])\n",
        "  \n",
        "\n",
        "  return model"
      ],
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jD5kbOmRmmVo"
      },
      "source": [
        "from kerastuner import RandomSearch\n",
        "from kerastuner.engine.hyperparameters import HyperParameters"
      ],
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Zl4hKLC1m5BX"
      },
      "source": [
        "tuner_search=RandomSearch(build_model,\n",
        "                          objective='val_accuracy',\n",
        "                          max_trials=5,directory='output',project_name=\"Mnist Fashion\") "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "w9nWsqC4oE12"
      },
      "source": [
        "tuner_search.search(train_images,train_labels,epochs=3,validation_aplit=0.1)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "obyTbb1pom8v"
      },
      "source": [
        "model=tuner_search.get_best_models(num_models=1)[0]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7NXf2BDJo7nS"
      },
      "source": [
        "model.summary()"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}
