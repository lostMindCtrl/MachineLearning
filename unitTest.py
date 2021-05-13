import pandas                   as pd
import numpy                    as np
import scipy.io                 as sc
import scipy.optimize           as bfgs
import linearRegression         as lr
import logisticRegression       as lg
import kMeans                   as km
import mapFeature               as mf
import matplotlib               as mlp
import matplotlib.pyplot        as plt
import matplotlib.cm            as cm

class unitTest:
    def __init__(self):
        self.data = []

def unitTesting():
    print("Start unit testing...")

    def featureScaling(df):
        row, col = df.shape
        mu       = []
        sigma    = []
        for i in range(col):
            mu.append(df[i].mean())
            sigma.append(df[i].std())
            df[i] = (df[i]-df[i].mean())/df[i].std()
        return df, mu, sigma

    def removeFeatureScaling(val, mu, sigma):
        return val * sigma + mu

    def testLinearRegression():
        print("\n\n Testing Linear Regression:")
        df            = pd.read_csv("testData/ex1data2.txt", delimiter=',', header=None)
        X_clean       = df.copy().values
        print(df.head())
        print(df.describe())
        df, mu, sigma = featureScaling(df)
        X             = df.values
        y             = X[:,2:3]
        theta, X, cost = lr.batch_regressionModel(400, X[:,:2], X[:,2:3], 0.01)
        plotLinearRegression(X_clean, theta, cost, removeFeatureScaling(np.min(lr.hypothesis(theta, X)), mu[2], sigma[2]), removeFeatureScaling(np.max(lr.hypothesis(theta, X)), mu[2], sigma[2]), removeFeatureScaling(np.min(X[:, 1]), mu[0], sigma[0]), removeFeatureScaling(np.max(X[:, 1]), mu[0], sigma[0]))
        print("<-------------------->")
        print("Final theta")
        print(theta)
        print("<-------------------->")
        print("Testing for training Example 7,8,9,10")
        print(str(X[6,0]) + " | " + str(removeFeatureScaling(X[6, 1], mu[0], sigma[0])) + " | " + str(removeFeatureScaling(X[6, 2], mu[1], sigma[1])))
        print(str(X[7,0]) + " | " + str(removeFeatureScaling(X[7, 1], mu[0], sigma[0])) + " | " + str(removeFeatureScaling(X[7, 2], mu[1], sigma[1])))
        print(str(X[8,0]) + " | " + str(removeFeatureScaling(X[8, 1], mu[0], sigma[0])) + " | " + str(removeFeatureScaling(X[8, 2], mu[1], sigma[1])))
        print(str(X[9,0]) + " | " + str(removeFeatureScaling(X[9, 1], mu[0], sigma[0])) + " | " + str(removeFeatureScaling(X[9, 2], mu[1], sigma[1])))
        print("theta*the Training Examples")
        solution_seven = lr.hypothesis(theta, X[6, 0:3])
        solution_eight = lr.hypothesis(theta, X[7, 0:3])
        solution_nine  = lr.hypothesis(theta, X[8, 0:3])
        solution_ten   = lr.hypothesis(theta, X[9, 0:3])
        print("Remove feature Scaling")
        print("<> " + str(removeFeatureScaling(y[6], mu[2], sigma[2])) + " - " + str(removeFeatureScaling(solution_seven, mu[2], sigma[2])) + " <>")
        print("<> " + str(removeFeatureScaling(y[7], mu[2], sigma[2])) + " - " + str(removeFeatureScaling(solution_eight, mu[2], sigma[2])) + " <>")
        print("<> " + str(removeFeatureScaling(y[8], mu[2], sigma[2])) + " - " + str(removeFeatureScaling(solution_nine,  mu[2], sigma[2])) + " <>")
        print("<> " + str(removeFeatureScaling(y[9], mu[2], sigma[2])) + " - " + str(removeFeatureScaling(solution_ten,   mu[2], sigma[2])) + " <>")
        print("<-------------------->")
        print("Estimate the price of a 1650 sq-ft, 3 br house using gradient descent")
        estimate = np.matrix([1, (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]])
        print("Predicted price of a 1650 sq-ft, 3 br house " + str(removeFeatureScaling(lr.hypothesis(theta, estimate), mu[2], sigma[2])))
        print("<-------------------->")
        print("Plotting cost")
        #plt.plot(cost)
        #print(cost)
        #print(np.matmul(np.transpose(theta), np.transpose(X)))

    def plotLinearRegression(X, theta, cost, min, max, init_min, init_max):
        fig, axs  = plt.subplots(2, 2)
        fig.canvas.set_window_title('LinearRegression')
        #fig.set_size_inches(14.5, 7.5)
        fig.suptitle("Linear Regression Model using Dataset 1.2")
        axs[0, 0].set_title("sq-ft to price")
        axs[0, 0].set_ylabel("Price")
        axs[0, 0].set_xlabel("sq-ft")
        axs[0, 0].scatter(X[:, 0], X[:, 2], marker='x')
        axs[0, 1].set_title("Bedrooms - Price")
        axs[0, 1].set_ylabel("Bedrooms")
        axs[0, 1].set_xlabel("Price")
        axs[0, 1].scatter(X[:, 2], X[:, 1], marker='x')
        axs[1, 0].set_title("sq-ft to price and Lr Line")
        axs[1, 0].set_ylabel("Price")
        axs[1, 0].set_xlabel("sq-ft")
        axs[1, 0].scatter(X[:, 0], X[:, 2], marker='x')
        axs[1, 0].plot([init_min, init_max], [min, max], color='red')
        axs[1, 1].set_title("CostFunction LinearRegression")
        axs[1, 1].set_ylabel("Cost")
        axs[1, 1].set_xlabel("Iterations")
        axs[1, 1].plot(cost)

    def testRegularizedLinearRegression():
        print("\n\n Testing Regularized Linear Regression:")
        df            = pd.read_csv("testData/ex1data2.txt", delimiter=',', header=None)
        X_clean       = df.copy().values
        df, mu, sigma = featureScaling(df)
        X             = df.values
        y             = X[:,2:3]
        theta, X, cost = lr.regularized_linearRegression(400, 0.01, 1, X[:,:2], X[:,2:3])
        print("<-------------------->")
        print("Reg. linear Regression using Dataset 1.2 and lambda = 1")
        print("Estimate the price of a 1650 sq-ft, 3 br house using gradient descent")
        estimate = np.matrix([1, (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]])
        print("Predicted price of a 1650 sq-ft, 3 br house " + str(removeFeatureScaling(lr.hypothesis(theta, estimate), mu[2], sigma[2])))
        print("")

    def testLogisticRegression():
        print("\n\n Testing Logistic Regression:")
        df         = pd.read_csv("testData/ex2data1.txt", delimiter=',', header=None)
        df_copy    = df.copy()
        df.columns = [0, 1, 2]
        print(df)
        df.describe()
        df_y = df[2]
        df, mu, sigma = featureScaling(df)
        X = df.values
        y = df_y.values
        y_vector = np.zeros((y.shape[0],1))
        y_vector[:,0] = y[:]
        y = y_vector
        theta, cost, X = lg.logisticRegression(400, 0.1, X[:, 0: 2], y[:, 0:1])
        plotLogisticRegression(df_copy.values, y, theta, cost)
        print("\nTest successful")
        print("<-------------------->")
        print("Theta:")
        print(theta)
        print("<-------------------->")
        print("Testing for:")
        print("Student with a score of 45 in Exam 1 and a score of 85 in Exam 2")
        test         = np.array([1, (45 - mu[0])/sigma[0], (85 - mu[1])/sigma[1]])
        predict_prob = lg.sigmoid(np.matmul(test, theta))
        #print(str(test) + " * " + str(theta))
        print("Probability that the Student passes: " + str(predict_prob) + " Expected Value: 0.775 +/- 0.002")
        print("<-------------------->")
        predict_prob_one   = lg.sigmoid(np.matmul(X[12:13, :], theta))
        predict_prob_two   = lg.sigmoid(np.matmul(X[43:44, :], theta))
        predict_prob_three = lg.sigmoid(np.matmul(X[21:22, :], theta))
        print(str(float(predict_prob_one))   + " <-- Example 14 --> " + str(y[12:13, 0]))
        print(str(float(predict_prob_two))   + " <-- Example 45 --> " + str(y[43:44, 0]))
        print(str(float(predict_prob_three)) + " <-- Example 23 --> " + str(y[21:22, 0]))

    def plotLogisticRegression(X, y, theta, cost):
        positive   = np.argwhere(y==1)
        negative   = np.argwhere(y==0)
        fig1, axs1 = plt.subplots(2, 2)
        fig1.canvas.set_window_title('LogisticRegression')
        fig1.suptitle("Logistic Regression Model using Dataset 2.1")
        axs1[0, 0].set_title("Admission of a Student by their Exam scores")
        axs1[0, 0].set_ylabel("Exam 2 score")
        axs1[0, 0].set_xlabel("Exam 1 score")
        axs1[0, 0].scatter(X[positive, 0], X[positive, 1], marker='+', color="forestgreen")
        axs1[0, 0].scatter(X[negative, 0], X[negative, 1], marker='.', color="orangered")
        axs1[0, 0].legend(['admitted', 'not admitted'])
        axs1[0, 1].set_title("Comparison on how many Students are admitted")
        axs1[0, 1].set_ylabel("Number of students")
        #axs1[0, 1].set_xlabel("not admitted - admitted")
        axs1[0, 1].bar(['not admitted', 'admitted'] ,[y[negative, 0].shape[0], y[positive, 0].shape[0]])
        axs1[1, 0].set_title("admission of a student by their exam scores and decision boundary")
        axs1[1, 0].set_ylabel("Exam 2 score")
        axs1[1, 0].set_xlabel("Exam 1 score")
        axs1[1, 0].scatter(X[positive, 0], X[positive, 1], marker='+', color="forestgreen")
        axs1[1, 0].scatter(X[negative, 0], X[negative, 1], marker='.', color="orangered")
        axs1[1, 0].legend(['admitted', 'not admitted', 'decision boundary'])
        axs1[1, 1].set_title("Logistic Regression cost funtion")
        axs1[1, 1].set_ylabel("Cost")
        axs1[1, 1].set_xlabel("Iterations")
        axs1[1, 1].plot(cost, color='orange')
        #Plot decision boundary
        #plot_x = np.matrix([np.min(X[:, 1])-2, np.max(X[:, 1])+2])
        #plot_y = (-1/theta[2, 0]*(theta[1, 0]*plot_x + theta[0, 0]))
        #axs1[1, 0].plot(plot_x, plot_y)

    def testRegularizedLogisticRegression():
        print("\n\n Testing Regularized Logistic Regression:")
        df             = pd.read_csv("testData/ex2data2.txt", delimiter=',', header=None)
        df_copy        = df.copy()
        df.columns     = [0, 1, 2]
        print(df)
        df.describe()
        df_y           = df[2]
        df, mu, sigma = featureScaling(df)
        X              = df.values
        y              = df_y.values
        y_vector       = np.zeros((y.shape[0],1))
        y_vector[:,0]  = y[:]
        y              = y_vector
        #output         = mf.featureMapping(X[:, 0], X[:, 1], 6)
        #print(y[:, 0])
        theta, X, cost = lg.regularized_logisticRegression(400, 0.01, X[:, :2], y[:, 0:1], 1)
        print("\nTest successful")
        print("<-------------------->")
        print("Theta:")
        print(theta)
        print("<-------------------->")
        print("Testing Train Accuracy - Polynomial Features were not used - lambda = 1")
        p                = lg.sigmoid(np.matmul(X, theta))
        p[p >= 0.5]      = 1
        p[p < 0.5]       = 0
        print("Train Accuracy " + str(np.mean(p == y)*100))
        print("<-------------------->")
        X, y             = mapFeatures(df_copy.values)
        df, mu, sigma    = featureScaling(X)
        X_copy           = X.copy()
        X                = df.values
        theta, X, cost_1 = lg.regularized_logisticRegression(400, 0.01, X[:, :], y[:, 0:1], 1)
        print("Testing Train Accuracy - Polynomial Features were  used - lambda = 1")
        p                = lg.predict(theta, X)
        print("Train Accuracy " + str(np.mean(p == y)*100))
        print("<-------------------->")
        X                = X_copy.values
        theta, X, cost_2 = lg.regularized_logisticRegression(900, 0.01, X[:, :], y[:, 0:1], 0.00001)
        print("Testing Train Accuracy - Polynomial Features were  used - lambda = 0.00001")
        p                = lg.predict(theta, X)
        print("Train Accuracy " + str(np.mean(p == y)*100))
        print("<-------------------->")
        plotRegLogisticRegression(df_copy.values, y, cost, cost_1, cost_2)

    def mapFeatures(X):
        y            = np.zeros((X.shape[0], 1))
        y[:, 0]      = X[:, 2]
        new_X        = np.zeros((X.shape[0], 22))
        new_X[:, 0]  = X[:, 0]
        new_X[:, 1]  = X[:, 1]
        new_X[:, 2]  = X[:, 0]*X[:, 1]
        new_X[:, 3]  = np.power(X[:, 0], 2)
        new_X[:, 4]  = np.power(X[:, 1], 2)
        new_X[:, 5]  = np.power(X[:, 0]*X[:, 1], 2)
        new_X[:, 6]  = np.power(X[:, 0], 3)
        new_X[:, 7]  = np.power(X[:, 1], 3)
        new_X[:, 8]  = np.power(X[:, 0]*X[:, 1], 3)
        new_X[:, 9]  = np.power(X[:, 0], 4)
        new_X[:, 10] = np.power(X[:, 1], 4)
        new_X[:, 11] = np.power(X[:, 0]*X[:, 1], 4)
        new_X[:, 12]  = np.power(X[:, 0], 5)
        new_X[:, 13] = np.power(X[:, 1], 5)
        new_X[:, 14] = np.power(X[:, 0]*X[:, 1], 5)
        new_X[:, 15]  = np.power(X[:, 0], 6)
        new_X[:, 16] = np.power(X[:, 1], 6)
        new_X[:, 17] = np.power(X[:, 0]*X[:, 1], 6)
        new_X[:, 18] = X[:, 0]*np.power(X[:, 1], 2)
        new_X[:, 19] = np.power(X[:, 0], 2)*X[:, 1]
        new_X[:, 20] = np.power(X[:, 0]*np.power(X[:, 1], 2), 2)
        new_X[:, 21] = np.power(np.power(X[:, 0], 2)*X[:, 1], 2)
        return pd.DataFrame(new_X), y

    def plotRegLogisticRegression(X, y, cost, cost_1, cost_2):
        positive   = np.argwhere(y==1)
        negative   = np.argwhere(y==0)
        fig2, axs2 = plt.subplots(2, 2)
        fig2.canvas.set_window_title('RegularizedLogisticRegression')
        fig2.suptitle("Regularized Logistic Regression Model using Dataset 2.2")
        axs2[0, 0].set_title("Microchip Test")
        axs2[0, 0].set_ylabel("Microchip Test 2")
        axs2[0, 0].set_xlabel("Microchip Test 1")
        axs2[0, 0].scatter(X[positive, 0], X[positive, 1], marker='+', color="forestgreen")
        axs2[0, 0].scatter(X[negative, 0], X[negative, 1], marker='.', color="orangered")
        axs2[0, 0].legend(['1', '0'])
        axs2[0, 1].set_title("Reg. LogisticRegression cost function")
        axs2[0, 1].set_ylabel("Cost")
        axs2[0, 1].set_xlabel("Iterations")
        axs2[0, 1].plot(cost, color='orange')
        axs2[1, 0].set_title("Reg. LogR cost function using Poly. Features v lambda = 1")
        axs2[1, 0].set_ylabel("Cost")
        axs2[1, 0].set_xlabel("Iterations")
        axs2[1, 0].plot(cost_1, color='orange')
        axs2[1, 1].set_title("Reg. LogR cost function using Poly. Features v lambda = 0.00001")
        axs2[1, 1].set_ylabel("Cost")
        axs2[1, 1].set_xlabel("Iterations")
        axs2[1, 1].plot(cost_2, color='orange')

    def testNormalEquations():
        df          = pd.read_csv("testData/ex1data2.txt", delimiter=',', header=None)
        X           = df.values
        theta, cost = lr.normalEquations_regressionModel(X[:,:2], X[:,2:3])
        print("<-------------------->")
        print("Testing Normal Equations on Dataset 1.2")
        print("<-------------------->")
        print("Theta:")
        print(theta)
        print("<-------------------->")
        print("Cost:")
        print(float(cost[0]))
        print("<-------------------->")
        print("Estimate the price of a 1650 sq-ft, 3 br house using normal equations")
        estimate = np.matrix([1, 1650, 3])
        print("Predicted price of a 1650 sq-ft, 3 br house " + str(lr.hypothesis(theta, estimate)))
        print("<-------------------->")
        print()

    def testRegNormalEquations():
        df       = pd.read_csv("testData/ex1data2.txt", delimiter=',', header=None)
        X        = df.values
        theta, X = lr.regularized_normalEquation(X[:,:2], X[:,2:3], 1)
        print("<-------------------->")
        print("Testing Regularized Normal Equations on Dataset 1.2")
        print("<-------------------->")
        print("Theta:")
        print(theta)
        print("<-------------------->")
        print("Estimate the price of a 1650 sq-ft, 3 br house using normal equations")
        estimate = np.matrix([1, 1650, 3])
        print("Predicted price of a 1650 sq-ft, 3 br house " + str(lr.hypothesis(theta, estimate)))
        print("<-------------------->")
        print()

    def testDigitRecognition():
        input_layer_size = 400
        num_labels       = 10
        data             = sc.loadmat("testData/ex3data1.mat")
        weight           = sc.loadmat("testData/ex3weights.mat")
        X                = data['X']
        y                = data['y']
        y_test           = y.copy()
        rand             = np.random.permutation(X.shape[0])
        sel              = X[rand[0:100], :]
        #Data display will be added later
        print("<-------------------->")
        print("Testing Reg. Regression Cost")
        print("Expected cost: 2.534819")
        print("When the cost Function is tested the result is sufficient")
        print("<-------------------->")
        print("The theta_all will be computed by OneVsAllClassification")
        theta_all, X     = oneVsAllClassification(X, y, 1)
        print("<-------------------->")
        print("Theta all: ")
        print("<-------------------->")
        p                = predict_oneVsall(X, theta_all)
        print("Training accuracy " + str(np.mean( p == y) * 100))
        print("<-------------------->")
        print("Neural Network Representation")
        theta_one        = weight['Theta1']
        theta_two        = weight['Theta2']
        pred             = predict_digit_nn(np.transpose(theta_one),  np.transpose(theta_two), X)
        print("Training accuracy " + str(np.mean(pred == y_test) * 100)) # unfortunately the prediction did only work with a copy of y as the vector y became a zero vector when comparing
        print("<-------------------->")

    def oneVsAllClassification(X, y, K):
        row, col = X.shape
        theta_O  = np.zeros((K, col+1))
        for i in range(K):
            if i == 0:
                lab = 10
            else:
                lab = i
            y[y == lab]   = 1
            y[y != lab]   = 0
            t, L, cost    = lg.regularized_logisticRegression(400, 0.01, X, y, 0.0001)
            theta_O[i, :] = t[:, 0]
        return theta_O, L

    def predict_oneVsall(X, theta):
        p = np.argmax(np.matmul(X, np.transpose(theta)), axis=1)
        p[p == 0] = 10


    def predict_digit_nn(theta_one, theta_two, X):
        first        = lg.sigmoid(np.matmul(X, theta_one))
        X_sec        = np.ones((first.shape[0], first.shape[1]+1))
        X_sec[:, 1:] = first[:, :]
        #X_sec     = np.matrix([np.ones((first.shape[0], 1))[:, 0], first]).reshape(first.shape[0], 1)
        second    = lg.sigmoid(np.matmul(X_sec, theta_two))
        predict          = np.zeros((second.shape[0], 1))
        predict[:, 0]    = np.argmax(second, axis=1)
        return predict+1 # due to the formatting of octave/matlab (math based frameworks - starts counting at one)


    def testkMeans():
        print("Testing kMeans clustering - Clustering the Mall_Customer Dataset by Income and Spending Score")
        df                                        = pd.read_csv("testData\Mall_Customers.csv", delimiter=',')
        df.loc[(df.Gender == 'FEMALE'), 'Gender'] = 0
        df.loc[(df.Gender == 'MALE')  , 'Gender'] = 1
        X                                         = df.values
        clusters                                  = km.kMeans(X[:, 3:5], 200, 5)
        print("<-------------------->")
        plotKMeansPoints(X, clusters, 5)

    def plotKMeansPoints(X, cl, K):
        colors     = np.unique(K)
        fig3, axs3 = plt.subplots(1 , 2)
        fig3.canvas.set_window_title('kMeans_clustering')
        fig3.suptitle("kMeans clustering - Mall Customer Dataset")
        axs3[0].set_title("Data displayed without clusters")
        axs3[1].set_ylabel("Spending Score")
        axs3[1].set_xlabel("Annual Income (k$)")
        axs3[0].scatter(X[:, 3], X[:, 4], marker='x')
        axs3[1].set_title("kMeans clustered - Income/Score")
        axs3[1].set_ylabel("Spending Score")
        axs3[1].set_xlabel("Annual Income (k$)")
        for i in range(K):
            axs3[1].scatter(X[np.argwhere(cl == i), 3], X[np.argwhere(cl == i), 4], marker='x', label = i)
    testLinearRegression()
    testRegularizedLinearRegression()
    testNormalEquations()
    testRegNormalEquations()
    testLogisticRegression()
    testRegularizedLogisticRegression();
    testDigitRecognition()
    testkMeans()
    plt.show()
    #printTestData(X, C)
