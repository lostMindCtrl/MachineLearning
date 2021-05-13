import pandas            as pd
import numpy             as np
import seaborn           as sb
import matplotlib        as mlp
import matplotlib.pyplot as plt
import linearRegression  as lr
import unitTest          as u


def main():
    print("<-------------------->")
    print("In the following project the assignments of the four weeks of the coursera course machine learning by andrew ng are displayed and tested.")
    print("Additionally, to the first four week Assigments the kMeans algorithm was implemented and tested with a dataset from kaggle (\"Mall_Customer.csv\").")
    print("Aforementioned algorithm needs the number of clusters to find in order to work. To decide which number of clusters is optimal, the elbowMethod will be utilized.")
    print("Lastly, a quick data analysis and prediction will be done using the famous CaliforniaHousing dataset from kaggle.")
    print("The project should depict my (Calvin Bialek) interest in machine and deep learning as well as data analysis and is intended to be used for educational purposes only!")
    print("The Project will first show the data plots of the unitTest class and then those  of the main class will be plotted")
    print("Project:")
    print("<-------------------->")
    print("\n\n")
    u.unitTesting()
    print("\n\n")
    Analysis()

def Analysis():

    def loadDataset():
        df          = pd.read_csv("housing.csv")
        row, col    = df.shape
        print("The given Data has " + str(col) + " features and " + str(row) + " data rows \n")
        print("Dataframe head: ")
        print(df.head())
        print("\nDataframe tail:")
        print(df.tail())
        print("\nDataframe Overview: ")
        print(df.describe())
        print("\n Datatypes in the dataframe: ")
        print(df.dtypes)
        df.dropna(axis=0, inplace = True)
        df_mod                    = df.copy()
        ocean_proximity_mapping   = {'INLAND': 0, 'NEAR BAY': 1, 'NEAR OCEAN': 2 ,'<1H OCEAN': 3, 'ISLAND': 4}
        df_mod                    = df_mod.replace({'ocean_proximity': ocean_proximity_mapping})
        df_mod.columns            = [0,1,2,3,4,5,6,7,8,9]
        df_mod[9]                 = df_mod[9].astype("float64")
        return df_mod, df

    def prepareSets(df):
        row, col = df.shape
        #Data Normalization
        for i in range(col):
            df[i] = df[i]/df[i].max()
        np_df                   = np.zeros((row, col))
        np_df                   = df.values
        split                   = int((row*3)/4);
        trainings_set           = np_df[:split,:]
        development_set         = np_df[split:,:]
        y_trainings_set         = np.zeros((split, 1))
        y_trainings_set[:, 0]   = trainings_set[:, 8]
        x_trainings_set         = np.delete(trainings_set, 8, axis=1)
        y_development_set       = np.zeros((row - split, 1))
        y_development_set[:, 0] = development_set[:, 8]
        x_development_set       = np.delete(development_set, 8, axis=1)
        return x_trainings_set, y_trainings_set, x_development_set, y_development_set

    def dataVisualization(df):
        bins_median_income              = np.linspace(min(df["median_income"]), max(df["median_income"]), 4)
        group_names                     = ["Low", "Medium", "High"]
        df["median_income-binned"]      = pd.cut(df["median_income"], bins_median_income, labels=group_names, include_lowest=True)
        bins_median_house_value         = np.linspace(min(df["median_house_value"]), max(df["median_house_value"]), 4)
        df["median_house_value-binned"] = pd.cut(df["median_house_value"], bins_median_house_value, labels=group_names, include_lowest=True)
        #testing the data visualization
        fig, axs = plt.subplots(2, 3)
        df_orig  = df.copy()
        df       = df.sample(100)
        print(df)
        print("plotting Data Graphs")
        fig.canvas.set_window_title('CaliforniaHousingData')
        fig.suptitle("CaliforniaHousing Dataset - data visualization")
        axs[0, 0].set_title("house_median_value in comparison to median_income")
        axs[0, 0].set_xlabel("median_income")
        axs[0, 0].set_ylabel("median_house_value")
        axs[0, 0].scatter(df["median_income"], df["median_house_value"], marker='x', color='red')
        axs[1, 0].set_title("house_median_value in comparison to total_bedrooms")
        axs[1, 0].set_xlabel("total_bedrooms")
        axs[1, 0].set_ylabel("median_house_value")
        axs[1, 0].scatter(df["total_bedrooms"], df["median_house_value"], marker='x', color='red')
        axs[0, 1].set_title("Location of the House Areas and their median house values")
        axs[0, 1].set_xlabel("longitude")
        axs[0, 1].set_ylabel("latitude")
        axs[0, 1].scatter(df_orig["longitude"], df_orig["latitude"], marker='x', c=df_orig["median_house_value"])
        axs[1, 1].set_title("Longitude*Latitude Graph")
        axs[1, 1].set_xlabel("logitude*latitude")
        axs[1, 1].set_ylabel("median_house_value")
        axs[1, 1].scatter((df["longitude"]*df["latitude"]), df["median_house_value"])
        axs[0, 2].hist(df["median_income-binned"])
        axs[0, 2].set_title("median_income_types")
        axs[0, 2].set_xlabel("median_income")
        axs[0, 2].set_ylabel("count")
        axs[1, 2].hist(df["median_house_value-binned"])
        axs[1, 2].set_title("median_value_types")
        axs[1, 2].set_xlabel("median_house_value")
        axs[1, 2].set_ylabel("count")
        plt.show()

    def dataPredictions(X_train, X_test, y_train, y_test):
        print("training and testing machine learning model")
        print("Shape of training_set -  X_train: " + str(X_train.shape) + ", y_train: " + str(y_train.shape))
        print("Shape of testing_set - X_test: "  + str(X_test.shape)  + ", y_test: "  + str(y_test.shape))
        print("\nLinear regression model ")
        print("Training without longitude and latitude (no polynomial features) and predicting median_house_value")
        theta_std, X_train, cost_std = lr.batch_regressionModel(400, X_train[:, 2:], y_train[:, :], 0.01)
        print("Testing with the Training set")
        print("Test Values:")
        print(X_test[:5,:])
        print("Comparing the Results (Original - Prediction)")
        print(X_test.shape[1])
        test_one                     = np.ones((X_test.shape[0], X_test.shape[1]-1))
        test_one[:, 1:]              = X_test[:,2:]
        print(str(y_test[:5, :]) + " <===> " + str(lr.hypothesis(theta_std, test_one[:5, :])))
        fig1, axs1 = plt.subplots(1, 1)
        fig1.canvas.set_window_title('CaliforniaHousingDataPredictions')
        fig1.suptitle("CaliforniaHousing Dataset - prediction cost")
        axs1.plot(cost_std)

    df, df_std                     = loadDataset()
    train_x, train_y, dev_x, dev_y = prepareSets(df)
    dataPredictions(train_x, dev_x, train_y, dev_y)
    dataVisualization(df_std)

if __name__ == "__main__":
    main()
    
