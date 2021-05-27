{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled6.ipynb",
      "provenance": [],
      "collapsed_sections": []
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
        "id": "WxEfXUyGb3uC",
        "outputId": "be165c2f-311c-4260-8bc8-38832d4f6985"
      },
      "source": [
        "!pip install keras-tuner"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: keras-tuner in /usr/local/lib/python3.7/dist-packages (1.0.2)\n",
            "Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (0.16.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (1.19.5)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (2.23.0)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (1.4.1)\n",
            "Requirement already satisfied: tabulate in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (0.8.9)\n",
            "Requirement already satisfied: colorama in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (0.4.4)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (20.9)\n",
            "Requirement already satisfied: terminaltables in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (3.1.0)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (4.41.1)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.7/dist-packages (from keras-tuner) (0.22.2.post1)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (2.10)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (1.24.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (2020.12.5)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->keras-tuner) (3.0.4)\n",
            "Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->keras-tuner) (2.4.7)\n",
            "Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->keras-tuner) (1.0.1)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IW_rTztscOj7"
      },
      "source": [
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "import numpy as np\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eQyGVrOfcnt9"
      },
      "source": [
        "fashion_mnist=keras.datasets.fashion_mnist"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JEY_5WThc2Ul"
      },
      "source": [
        "(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-OmtYR8LdS9D"
      },
      "source": [
        "train_images=train_images/255.0 \n",
        "test_images=test_images/255.0 "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "k7ZljJNHdjYC",
        "outputId": "5448fe78-5b47-44bb-f7fa-4e7ae40eaaca"
      },
      "source": [
        "train_images[0].shape "
      ],
      "execution_count": null,
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
        "id": "U5ni-RzBdqZW"
      },
      "source": [
        "train_images = train_images.reshape(len(train_images),28,28,1) \n",
        "test_images = test_images.reshape(len(test_images),28,28,1) "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "cellView": "code",
        "id": "4K6Qx0D_eQQA"
      },
      "source": [
        "\n",
        "def build_model(hp):\n",
        "  model = keras.Sequential([\n",
        "    keras.layers.conv2D(\n",
        "        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),\n",
        "        kernel_size=hp.choice('conv_1_kernel', values = [3,5]),\n",
        "        activation='relu',\n",
        "        input_shape=(28,28,1)\n",
        "    ),\n",
        "    keras.layers.conv2D(\n",
        "        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),\n",
        "        kernel_size=hp.choice('conv_2_kernel', value = [3,5]),\n",
        "        activation='relu'\n",
        "    ),\n",
        "    keras.layers.Flatten(),\n",
        "    keras.layer.Dense(\n",
        "        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),\n",
        "        activation='relu'\n",
        "    ),\n",
        "    keras.layers.Dense(10,activation='softmax')#output layer\n",
        "  ])\n",
        "\n",
        "  model.compile(optimizer=keras.optimizers.Adam(hp.choice('learning_rate', values=[1e-2, 1e-3])), \n",
        "              loss='sparse_categorical_crossentropy',\n",
        "              metrics=['accuracy'])\n",
        "  \n",
        "  return model "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "60VAVfT4kF7h"
      },
      "source": [
        "from kerastuner import RandomSearch \n",
        "from kerastuner.engine.hyperparameters import HyperParameters "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7YHFnl6Skcw8"
      },
      "source": [
        "tuner_search=RandomSearch(build_model,\n",
        "                          objective='val_accuracy',\n",
        "                          max_trials=5,directory='output',project_name=\"Mnist Fashion\") \n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3cVM2B0j4DN4"
      },
      "source": [
        "tuner_search.search(train_images,train_labels,epochs=3,validation_split=0.1) "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vtSuNMiA48bb"
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
        "id": "0dVrnG_F5UdX"
      },
      "source": [
        "model.summary() "
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}
