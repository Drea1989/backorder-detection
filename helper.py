# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:07:07 2017

@author: andre
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from time import time
import scipy
import matplotlib.patches as mpatches
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.decomposition import PCA

#plot stats
def stats_plots_label(df,labels):
    means1 = df[labels==0].mean(axis=1)
    medians1 = df[labels==0].median(axis=1)
    std1 = df[labels==0].std(axis=1)
    maxval1 = df[labels==0].max(axis=1)
    minval1 = df[labels==0].min(axis=1)
    skew1 = df[labels==0].skew(axis=1)
    means2 = df[labels==1].mean(axis=1)
    medians2 = df[labels==1].median(axis=1)
    std2 = df[labels==1].std(axis=1)
    maxval2 = df[labels==1].max(axis=1)
    minval2 = df[labels==1].min(axis=1)
    skew2 = df[labels==1].skew(axis=1)
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(231)
    ax.hist(means1,alpha=0.8,bins=50,color='b',normed=True,range=(-250,250))
    ax.hist(means2,alpha=0.8,bins=50,color='r',normed=True,range=(-250,250))
    ax.get_legend()
    ax.set_xlabel('Mean Intensity')
    ax.set_ylabel('Num. of Stars')
    ax = fig.add_subplot(232)
    ax.hist(medians1,alpha=0.8,bins=50,color='b',normed=True,range=(-0.1,0.1))
    ax.hist(medians2,alpha=0.8,bins=50,color='r',normed=True,range=(-0.1,0.1))
    ax.get_legend()

    ax.set_xlabel('Median Intensity')
    ax.set_ylabel('Num. of Stars')
    ax = fig.add_subplot(233)    
    ax.hist(std1,alpha=0.8,bins=50,normed=True,color='b',range=(0,4000))
    ax.hist(std2,alpha=0.8,bins=50,normed=True,color='r',range=(0,4000))
    ax.get_legend()

    ax.set_xlabel('Intensity Standard Deviation')
    ax.set_ylabel('Num. of Stars')
    ax = fig.add_subplot(234)
    ax.hist(maxval1,alpha=0.8,bins=50,normed=True,color='b',range=(-10000,50000))
    ax.hist(maxval2,alpha=0.8,bins=50,normed=True,color='r',range=(-10000,50000))
    ax.get_legend()

    ax.set_xlabel('Maximum Intensity')
    ax.set_ylabel('Num. of Stars')
    ax = fig.add_subplot(235)
    ax.hist(minval1,alpha=0.8,bins=50,normed=True,color='b',range=(-50000,10000))
    ax.hist(minval2,alpha=0.8,bins=50,normed=True,color='r',range=(-50000,10000))
    ax.get_legend()

    ax.set_xlabel('Minimum Intensity')
    ax.set_ylabel('Num. of Stars')
    ax = fig.add_subplot(236)
    ax.hist(skew1,alpha=0.8,bins=50,normed=True,color='b',range=(-40,60))
    ax.hist(skew2,alpha=0.8,bins=50,normed=True,color='r',range=(-40,60)) 
    ax.get_legend()

    ax.set_xlabel('Intensity Skewness')
    ax.set_ylabel('Num. of Stars')
    stats_plots_label(log_data)
    plt.show()

#visualise class imbalance
def vs_class_imbalance(X,y,X_resampled,y_resampled):
    # Instanciate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=2)
    # Fit and transform x to visualise inside a 2D feature space
    X_vis = pca.fit_transform(X)

    # Apply SMOTE + Tomek links
    X_res_vis = pca.transform(X_resampled)

    # Two subplots, unpack the axes array immediately
    f , (ax1, ax2) = plt.subplots(1,2, figsize =(24,24))

    c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0",
                     alpha=0.5)
    c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1",
                     alpha=0.5)
    ax1.set_title('Original set')

    ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
                label="Class #0", alpha=0.5)
    ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
                label="Class #1", alpha=0.5)
    ax2.set_title('SMOTE + Tomek')

    # make nice plotting
    for ax in (ax1, ax2):
        ax.fill(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        #ax.set_xlim([-6, 8])
        #ax.set_ylim([-6, 6])

    plt.figlegend((c0, c1), ('Class #0', 'Class #1'), loc='lower center',
                  ncol=2, labelspacing=0.)
    plt.tight_layout(pad=3)
    plt.show()

def find_outliers(data):
        indices = []
        # for feature in norm_data.keys():
        for feature in data.keys():
            # TODO: Calculate Q1 (25th percentile of the data) for the given feature
            Q1 = np.percentile(data[feature], 25)

            # TODO: Calculate Q3 (75th percentile of the data) for the given feature
            Q3 = np.percentile(data[feature], 75)

            # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
            step = (Q3 - Q1) * 1.5

            # Display the outliers
            # print "Data points considered outliers for the feature '{}':".format(feature)
            # display(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))])
            indices.extend(data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))].index.tolist())

        # look for indices that repeats at least twice
        from collections import Counter
        cnt = Counter()
        for index in indices:
            cnt[index] += 1
        return cnt


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    from sklearn.metrics import fbeta_score, accuracy_score
    results = {}

    # Fit the learner to the training data using slicing with 'sample_size'
    start = time()  # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()  # Get end time
    results['train_time'] = end - start

    # Get the predictions on the test set,
    # then get predictions on the first 300 training samples
    start = time()  # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()  # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=20)

    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=20)

    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)

    # Return the results
    return results


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = plt.subplots(2, 3, figsize=(11, 7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000', '#00A0A0', '#00A000']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                # Creative plot code
                ax[j / 3, j % 3].bar(i + k * bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j / 3, j % 3].set_xticks([0.45, 1.45, 2.45])
                ax[j / 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j / 3, j % 3].set_xlabel("Training Set Size")
                ax[j / 3, j % 3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    plt.legend(handles=patches, bbox_to_anchor=(-.80, 2.53), \
               loc='upper center', borderaxespad=0., ncol=3, fontsize='x-large')

    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10)
    plt.tight_layout()
    plt.show()

def vs_frontiers(good_data,good_lables,outliers):
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.font_manager
    from sklearn import svm
    pca = PCA(n_components=2, random_state = 42)
    #X_train, X_test, y_train, y_test = train_test_split(good_data[good_lables ==0].sample(10000), good_lables[good_lables ==0].samlep(10000), test_size=0.4, random_state=42)
    #xx, yy = np.meshgrid(good_data,500,sparse=True)
    xx, yy = np.meshgrid(np.linspace(-20, 20, 600), np.linspace(-20,20, 600))

    X_train = pca.fit_transform(good_data[good_lables == 0].sample(40000))
    X_test = pca.transform(good_data[good_lables == 0].sample(10634))
    X_outliers = pca.transform(good_data[good_lables == 1])
    X_TrainOUTliers = pca.transform(outliers[good_lables == 0])

    # fit the model
    clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.045).fit(X_train)
    #predict
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers =clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(20,20))
    plt.title("Novelty Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    s = 30
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s)
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s)
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s)
    d = plt.scatter(X_TrainOUTliers[:, 0], X_TrainOUTliers[:, 1], c='red', s=s)
    plt.axis('tight')
    plt.xlim((-5, 20))
    plt.ylim((-10, 10))
    plt.legend([a.collections[0], b1, b2, c],
               ["learned frontier", "training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "error train: %d/40000 ; errors novel regular: %d/10634 ; "
        "errors novel abnormal: %d/10634"
        % (n_error_train, n_error_test, n_error_outliers))
    plt.show()