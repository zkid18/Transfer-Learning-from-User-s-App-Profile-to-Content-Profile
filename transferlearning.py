import tensorflow as tf
import numpy as np
import csv
from preprocessing import datas

input,output,alltest,alluser = datas()
print len(input)
print len(output)

#=================== add one more layer and return the output of this layer =====================
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(10*tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs,Weights,biases


#===================the function of testing training accuracy and test accuracy===================
def accuracy(output,prediction):
    accuracy = np.zeros(len(output))
    for i in range(0, len(output)):
        index = np.argsort(-prediction[i])
        if len(np.nonzero(output[i])[0]) >= 5:
            for j in range(0, 5):
                if index[j] in np.argsort(-output[i])[:5]:
                    accuracy[i] = accuracy[i] + 0.2
        else:
            for j in range(0, len(np.nonzero(output[i])[0])):
                if np.nonzero(output[i])[0][j] in index[:5]:
                    accuracy[i] = accuracy[i] + 1 / float(len(np.nonzero(output[i])[0]))
    return np.mean(accuracy)


# ==============================label of data from source domain or target domain==============
id_label = np.zeros((9842,1))
for id in range(0,4697):
    id_label[id] = 1.0


# ========================training set and test set=====================================
num1 = int(5872*0.8+5145)
num2 = int(5872*0.8)
input_tr = np.vstack((input[:num2],input[5872:]))
output_tr = output[:num2]

input_tst = input[num2:5872]
output_tst = output[num2:5872]

# ================== define placeholder for inputs to network============================
xs = tf.placeholder(tf.float32, [None, 55])
cat = tf.placeholder(tf.float32, [None,150])
labels = tf.placeholder(tf.float32, [None, 1])


#==================project the data to a new space with 100 dimension=====================
data_layer1,Weights1,biases1 = add_layer(xs, 55, 100, activation_function=tf.nn.relu)
data_layer2,Weights2,biases2 = add_layer(data_layer1, 100, 100, activation_function=tf.nn.relu)


#=============================select the data from source domain==========================
datalabel = data_layer2[0:num2,:]

#=======================u========category classifier======================================
data_layer4,Weights4,biase4 = add_layer(datalabel, 100, 100)
data_layer5,Weights5,biase5 = add_layer(datalabel, 100, 150)
cat_prediction = tf.nn.softmax(data_layer5,1)

#=================================domain classifier=======================================
data_layer6,Weights6,biases6 = add_layer(data_layer2, 100, 1)
domain_prediction = tf.nn.sigmoid(data_layer6)

#=============================loss function and training step=============================
loss1 = 0.5*(tf.reduce_mean(tf.reduce_sum(tf.square(domain_prediction-labels)))+0.001*(tf.nn.l2_loss(Weights6)+tf.nn.l2_loss(Weights1)+tf.nn.l2_loss(Weights2)))
loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(cat_prediction-cat),1))+0.001*(tf.nn.l2_loss(Weights1)+tf.nn.l2_loss(Weights2)+tf.nn.l2_loss(Weights4)+tf.nn.l2_loss(Weights5))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
train_op1 = optimizer.minimize(-loss1, var_list=[Weights1,Weights2,biases1,biases2])
train_op2 = optimizer.minimize(loss1, var_list=[Weights6,biases6])
train_op3 = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss2)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    training_accuracy = []
    test_accuracy = []
    for i in range(400000):
        if i == 10000:
            break
        sess.run([train_op1,train_op2,train_op3], feed_dict={xs: input_tr, cat: output_tr, labels: id_label})
        if i%50 == 0 :
            print i
            print 'loss of domain classifier =',sess.run(loss1, feed_dict={xs: input_tr, cat: output_tr, labels: id_label})
            print 'loss of category classifier =',sess.run(loss2, feed_dict={xs: input_tr, cat: output_tr, labels: id_label})
            # print sess.run(domain_prediction, feed_dict={xs: input, cat: output, labels: id_label})
            prediction1 = sess.run(cat_prediction, feed_dict={xs: input_tst})
            prediction2 = sess.run(cat_prediction, feed_dict={xs: input_tr})
            print 'training accuracy =',accuracy(output_tst,prediction1)
            training_accuracy.append(accuracy(output_tst,prediction1))
            print 'test accuracy =',accuracy(output_tr,prediction2)
            test_accuracy.append(accuracy(output_tr,prediction2))
    print 'max training accuracy =',np.max(training_accuracy)
    print 'max test accuracy =', np.max(test_accuracy)
    fi = open('./prediction.txt',"w")
    prediction1 = sess.run(cat_prediction, feed_dict={xs: alltest})
    for t in range(len(prediction1)):
        fi.write(alluser[t]+","+str(prediction1[t]))
    fi.close()