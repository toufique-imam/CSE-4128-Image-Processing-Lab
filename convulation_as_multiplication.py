import numpy as np

def create_toeplitz(vec,col):
    row = vec.shape[0]
    out = np.zeros((row, col))
    #print(vec)
    k=0
    while(k<row):
        i=k
        j=0
        while(i<row and j<col):
            out[i,j]=vec[k]
            i=i+1
            j=j+1
        k=k+1
    return out

def create_doublyLinkedMatrix(toeplitz_list,dl_indices):
    toeplitz_shape = toeplitz_list[0].shape
    h = toeplitz_shape[0] * dl_indices.shape[0]
    w = toeplitz_shape[1] * dl_indices.shape[1]
    dbmatrix = np.zeros((h,w))

    b_h = toeplitz_shape[0]
    b_w = toeplitz_shape[1]

    for i in range(dl_indices.shape[0]):
        for j in range(dl_indices.shape[1]):
            start_i = int(i*b_h)
            start_j = int(j*b_w)
            end_i = int(start_i+b_h)
            end_j = int(start_j+b_w)
            dbmatrix[start_i:end_i,start_j:end_j] = toeplitz_list[dl_indices[i,j]-1]
    return dbmatrix

def matrix_to_vector(matrix):
    row ,col = matrix.shape
    output = []
    for i in range(row-1,-1,-1):
        for j in range(0,col):
            output.append(matrix[i][j])
    return np.array(output)

def vector_to_maxtrix(vector,m_shape):
    row = m_shape[0]
    col = m_shape[1]
    output = np.zeros(m_shape)
    i = row-1
    j=0
    for k in range(len(vector)):
        output[i,j]=vector[k]
        j=j+1
        if(j>=col):
            j=0
            i=i-1
    return output
    
def convulation_as_matrix_multiplication(I,F):
    '''
    input is m1*n1 and filter is m2*n2 ;
    convulation will be (m1+m2-1)*(n1+n2-1)
    '''
    I_row_num , I_col_num = I.shape
    F_row_num , F_col_num = F.shape

    output_row_num = I_row_num+F_row_num-1
    output_col_num = I_col_num+F_col_num-1

    #print('output dimension :',output_row_num,output_col_num)

    F_zero_padded = np.pad(F,((output_row_num-F_row_num,0),(0,output_col_num-F_col_num)),'constant',constant_values = 0)
    #print('F_Zero_Padded: ',F_zero_padded)

    toeplitz_list = []
    toeplitz_indices=[]
    idxnow = 1

    for i in range(F_zero_padded.shape[0]-1,-1,-1):
        if(idxnow > F.shape[0]+1):
            toeplitz_indices.append(idxnow-1)
            continue
        tmp = F_zero_padded[i,:]
        toeplitz_list.append(create_toeplitz(tmp,I.shape[1]))
        toeplitz_indices.append(idxnow)
        idxnow=idxnow+1
    toeplitz_indices = np.array(toeplitz_indices, dtype=np.int)
    dl_indices = create_toeplitz(toeplitz_indices,I.shape[0])
    dl_indices = dl_indices.astype(np.int)
    dlmatrix = create_doublyLinkedMatrix(toeplitz_list,dl_indices)
    vectored_I = matrix_to_vector(I)
    # print(vectored_I)
    result_vector = np.matmul(dlmatrix,vectored_I)
    # print("result:\n",result_vector)
    result_matrix = vector_to_maxtrix(result_vector,(output_row_num,output_col_num))
    #print(result_matrix)
    return result_matrix

def convulation_mm(I,F):
    div_count = I.shape[0]
    #print(div_count)
    ans = 1
    for i in range(1,div_count):
        if(div_count%i==0):
            ans=i
    #print(i,I.shape)
    # return I
    div_count=div_count//ans
    I_row_num, I_col_num = I.shape
    F_row_num, F_col_num = F.shape

    output_row_num = I_row_num+F_row_num-1
    output_col_num = I_col_num+F_col_num-1
    I_split = np.array_split(I,div_count);
    res = []
    r1 = 0
    r2 = 0
    pad_r = F.shape[0]//2
    pad_c = F.shape[1]//2
    t = -1
    for Ix in I_split:
        print("sp shape",Ix.shape)
        outtmp = convulation_as_matrix_multiplication(Ix,F)
        r1 = r1 + outtmp.shape[0]
        r2 = r2 + outtmp.shape[1]
        ftmp = []
        for i in range(pad_r,outtmp.shape[0]-pad_r):
            tmp = []
            for j in range(pad_c, outtmp.shape[1]-pad_c):
                tmp.append(outtmp[i][j])
            ftmp.append(tmp)
        ftmp = np.array(ftmp)
        if(t==-1):
            t=ftmp.shape[0]
            res.append(ftmp)
        else:
            if(ftmp.shape[0]==t):
                res.append(ftmp)
                print("mm shape",ftmp.shape)
            else:
                print("dumped",ftmp.shape)
                continue
    res = np.array(res)
    print(res.shape)
    res = np.reshape(res,(res.shape[0]*res[0].shape[0],res[0].shape[1]))
    return res

from scipy import signal
if __name__ == "__main__":    
    I = np.array([[1, 2, 3], [4, 5, 6],[7,8,9]])
    print('I: ', I.shape)
    print(I)

    # filter
    F = np.array([[10, 20], [30, 40]])
    print('F: ', F.shape)
    print(F)
    res1 = convulation_as_matrix_multiplication(I,F)
    res = signal.convolve2d(I,F)
