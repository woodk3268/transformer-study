import numpy as np
from scipy.special import softmax

# step1 : represent the input
print("Step 1 : input : 3 inputs, d_model = 4")
x = np.array([[1.0, 0.0, 1.0, 0.0], # input 1
                [0.0, 2.0, 0.0, 2.0] # input2
                [1.0, 1.0, 1.0, 1.0]]) # input3

print(x)

# step2 : initializing the weight matrices

print("step 2 : weights 3 dimensions x d_model =4")
print("w_query")
w_query = np.array([[1,0,1],
                    [1,0,0],
                    [0,0,1],
                    [0,1,1]])
print(w_query)

print("w_key")
w_key = np.array([[0,0,1],
                [1,1,0],
                [0,1,0],
                [1,1,0]])
print(w_key)

print("w_value")
w_value = np.array([[0,2,0],
                    [0,3,0],
                    [1,0,3],
                    [1,1,0]])
print(w_value)

# step3 : matrix multiplication to obtain q,k and v
print("step 3 : matrix multiplication to obtain q,k,v")

print("queries: x * w_query")
Q = np.matmul(x, w_query)
print(Q)

print("keys : x * w_key")
K = np.matmul(x, w_key)
print(K)

print("values : x * w_value")
V = np.matmul(x,w_value)
print(V)

# step4 : scaled attention scores
print("step4 : scaled attention scores")
k_d = 1 # square root of k_d simplified to 1 for this example
attention_scores = (Q @ K.transpose())/k_d
print(attention_scores)

# step5 : scaled softmax attention scores for each vector
print("step5 : scaled softmax attention_scores for each vector")
attention_scores[0]=softmax(attention_scores[0])
attention_scores[1]=softmax(attention_scores[1])
attention_scores[2]=softmax(attention_scores[2])
print(attention_scores[0])
print(attention_scores[1])
print(attention_scores[2])

# step6 : the final attention representations
print("step 6 : attention value obtained by score1/k_d *V")
print(V[0])
print(V[1])
print(V[2])
print("Attention 1")
attention1=attention_scores[0].reshape(-1,1)
attention1=attention_scores[0][0]*V[0]
print(attention1)

print("Attention 2")
attention2=attention_scores[0][1]*V[1]
print(attention2)

print("Attention 3")
attention3=attention_scores[0][2]*V[2]
print(attention3)

# step7 : summing up the results
print("step 7 : summed the results to create the first line of the output matrix")
attention_input1 = attention1 + attention2 +attention3
print(attention_input1)

# step8 : steps 1 to 7 for all the inputs
print("step 8 : step 1 to 7 for inputs 1 to 3")
# we assume we have 3 results with learned weights (they were not trained in this example)
# we assume we are implementing the original transformer paper. we will have 3 results of 64 dimensions each
attention_head1 = np.random.random((3,64))
print(attention_head1)

# step9 : the output of the heads of the attention sublayer

print("Step 9: We assume we have trained the 8 heads of the attention sub-layer")
z0h1=np.random.random((3, 64))
z1h2=np.random.random((3, 64))
z2h3=np.random.random((3, 64))
z3h4=np.random.random((3, 64))
z4h5=np.random.random((3, 64))
z5h6=np.random.random((3, 64))
z6h7=np.random.random((3, 64))
z7h8=np.random.random((3, 64))
print("shape of one head",z0h1.shape,"dimension of 8 heads",64*8)

# step 10 : concatenation of the output of the heads


print("Step 10: Concatenation of heads 1 to 8 to obtain the original 8x64=512 output dimension of the model")
output_attention=np.hstack((z0h1,z1h2,z2h3,z3h4,z4h5,z5h6,z6h7,z7h8))
print(output_attention)



#@title Retrieve pipeline of modules and choose English to French translation
from transformers import pipeline
translator = pipeline("translation_en_to_fr")
#One line of code!
print(translator("It is easy to translate languages with transformers", max_length=40))