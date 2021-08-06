import tensorflow as tf
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
model=tf.keras.models.Sequential()
# noinspection PyInterpreter
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(512,activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy"
              ,metrics=["accuracy"])
model.fit(x_train,y_train,epochs=10)
val_loss,val_acc=model.evaluate(x_test,y_test)
print("loss : ",val_loss,"\naccuracy : ",val_acc)
model.save("0.model")
model0=tf.keras.models.load_model("1.model")





