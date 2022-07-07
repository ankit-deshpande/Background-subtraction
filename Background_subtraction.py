import numpy as np
import cv2 as cv

def norm_pdf(x,mean,sigma):
    return (1/(np.sqrt(2*3.14)*sigma))*(np.exp(-0.5*(((x-mean)/sigma)**2)))


cap = cv.VideoCapture("umcp.mpg")
ret,col_frame = cap.read()
frame = cv.cvtColor(col_frame,cv.COLOR_BGR2GRAY)

row,col = frame.shape

mean = np.zeros([3,row,col],np.float64)
mean[1,:,:] = frame

variance = np.zeros([3,row,col],np.float64)
variance[:,:,:] = 400

omega = np.zeros([3,row,col],np.float64)
omega[0,:,:],omega[1,:,:],omega[2,:,:] = 0,0,1

omega_by_sigma = np.zeros([3,row,col],np.float64)

foreground = np.zeros([row,col],np.uint8)
background = np.zeros([row,col],np.uint8)
background=frame

#initialising T and alpha
alpha = 0.8
T = 0.3

# fourcc = cv.VideoWriter_fourcc(*'MJPG')
# out = cv.VideoWriter('background.avi',fourcc, 20, (352,240),isColor=False)

while cap.isOpened():
    ret,frame = cap.read()
    
    if cv.waitKey(1) & 0xFF == 27 or ret == False:
        break
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    frame_gray = frame_gray.astype(np.float64)
    
    #Variance is negative sometime so we will put some value if var is near to zero
    variance[0][np.where(variance[0]<1)] = 10
    variance[1][np.where(variance[1]<1)] = 5
    variance[2][np.where(variance[2]<1)] = 1
    

    #calulating standard deviation
    sigma1 = np.sqrt(variance[0])
    sigma2 = np.sqrt(variance[1])
    sigma3 = np.sqrt(variance[2])

    # getting values for the inequality test to get indexes of fitting indexes
    compare_val_1 = cv.absdiff(frame_gray,mean[0])
    compare_val_2 = cv.absdiff(frame_gray,mean[1])
    compare_val_3 = cv.absdiff(frame_gray,mean[2])

    value1 = 2.5 * sigma1
    value2 = 2.5 * sigma2
    value3 = 2.5 * sigma3
    
    # Finding those indices where a particular pixel values fits at least one of the gaussian
    gauss_fit_index1 = np.where(compare_val_1 <= value1)
    gauss_not_fit_index1 = np.where(compare_val_1 > value1)

    gauss_fit_index2 = np.where(compare_val_2 <= value2)
    gauss_not_fit_index2 = np.where(compare_val_2 > value2)

    gauss_fit_index3 = np.where(compare_val_3 <= value3)
    gauss_not_fit_index3 = np.where(compare_val_3 > value3)
    
    
    match_index = np.zeros([row,col])
    match_index[gauss_fit_index1] = 1
    match_index[gauss_fit_index2] = 1
    match_index[gauss_fit_index3] = 1
    not_match_index = np.where(match_index == 0)
    
    
    #updating variance and mean value of the matched indices of all three gaussians
    rho = alpha * norm_pdf(frame_gray[gauss_fit_index1], mean[0][gauss_fit_index1], sigma1[gauss_fit_index1])
    constant = rho * ((frame_gray[gauss_fit_index1] - mean[0][gauss_fit_index1]) ** 2)
    mean[0][gauss_fit_index1] = (1 - rho) * mean[0][gauss_fit_index1] + rho * frame_gray[gauss_fit_index1]
    variance[0][gauss_fit_index1] = (1 - rho) * variance[0][gauss_fit_index1] + constant
    omega[0][gauss_fit_index1] = (1 - alpha) * omega[0][gauss_fit_index1] + alpha
    omega[0][gauss_not_fit_index1] = (1 - alpha) * omega[0][gauss_not_fit_index1]

    rho = alpha * norm_pdf(frame_gray[gauss_fit_index2], mean[1][gauss_fit_index2], sigma2[gauss_fit_index2])
    constant = rho * ((frame_gray[gauss_fit_index2] - mean[1][gauss_fit_index2]) ** 2)
    mean[1][gauss_fit_index2] = (1 - rho) * mean[1][gauss_fit_index2] + rho * frame_gray[gauss_fit_index2]
    variance[1][gauss_fit_index2] = (1 - rho) * variance[1][gauss_fit_index2] + rho * constant
    omega[1][gauss_fit_index2] = (1 - alpha) * omega[1][gauss_fit_index2] + alpha
    omega[1][gauss_not_fit_index2] = (1 - alpha) * omega[1][gauss_not_fit_index2]

    rho = alpha * norm_pdf(frame_gray[gauss_fit_index3], mean[2][gauss_fit_index3], sigma3[gauss_fit_index3])
    constant = rho * ((frame_gray[gauss_fit_index3] - mean[2][gauss_fit_index3]) ** 2)
    mean[2][gauss_fit_index3] = (1 - rho) * mean[2][gauss_fit_index3] + rho * frame_gray[gauss_fit_index3]
    variance[2][gauss_fit_index3] = (1 - rho) * variance[2][gauss_fit_index3] + constant
    omega[2][gauss_fit_index3] = (1 - alpha) * omega[2][gauss_fit_index3] + alpha
    omega[2][gauss_not_fit_index3] = (1 - alpha) * omega[2][gauss_not_fit_index3]
    
    # updating least probable gaussian for those pixel values which do not match any of the gaussian
    mean[0][not_match_index] = frame_gray[not_match_index]
    variance[0][not_match_index] = 200
    omega[0][not_match_index] = 0.1

    # normalise omega
    sum = np.sum(omega,axis=0)
    omega = omega/sum

    #finding omega by sigma for ordering of the gaussian
    omega_by_sigma[0] = omega[0] / sigma1
    omega_by_sigma[1] = omega[1] / sigma2
    omega_by_sigma[2] = omega[2] / sigma3

    # getting index order for sorting omega by sigma
    index = np.argsort(omega_by_sigma,axis=0)
    back_index0 = np.where(omega_by_sigma[0]>T)
    back_index1 = np.where(omega_by_sigma[1]>T)
    back_index2 = np.where(omega_by_sigma[2]>T)
    
    # from that index sorting mean,variance and omega
    mean = np.take_along_axis(mean,index,axis=0)
    variance = np.take_along_axis(variance,index,axis=0)
    omega = np.take_along_axis(omega,index,axis=0)
    
    frame_gray = frame_gray.astype(np.uint8)
    
    
    background[back_index1]=frame_gray[back_index1]
    background[back_index2]=frame_gray[back_index2]
    
    
    cv.imshow('frame_gray',frame_gray)
    cv.imshow('Background',background)
    cv.imshow('Foreground',cv.absdiff(frame_gray,background))
    # out.write(background)
    
cap.release()
# out.release()
cv.destroyAllWindows()

     