#   BADM 590 Analytics

2017-11-27 · Neal Davis

##  Processes in Data Analysis

We'll start off with the basic steps of data analysis.  This is the unsexy part of "data analytics".  Everyone loves to talk about data analytics but there's a lot of junk out there as well:  bad statistics, incomplete data, half-baked conclusions.

>  Q.  What is data analytics?
>  A.  Statistics on a Mac.


### Data Sources

Databases are commonly available as SQL databases.  Data sets can be available as `csv` files, or you may have to scrape data from a website or other form.

### Data Cleaning

Data are often quite messy.  Some issues include `NONE`, `null`, `0`, `NaN`, `-9999`, `None`, and other values.

### Data Integration

Data integration refers to bringing together data from disparate sources, such as weather data and hydrological flow data.  These often are not available at the same sampling frequency or in incompatible formats.

### Analysis & Statistical Methods

Tasks in data analytics range from high-level framings to nitty-gritty implementations.  Implementations normally deal with either _classification_ or _regression_.

-   _Classification_:  construction of a decision basis
-   _Regression_:  construction of a relationship

The term "data mining" typically refers to extractive analytics.  
Machine learning, in contrast, generally describes predictive methods which frequently train on data sets in order to improve their classification or regression ability.

### Reporting

### Knowledge-Driven Discovery Process

The Knowledge-Driven Discovery (KDD) process formally consists of an expansion of my suggested process:

1.  **Definition**.  Understand the application domain and identify the goal of the KDD process from the customer’s perspective.
2.  **Selection**.  Select a target data set or subset of data samples on which discovery is be performed.
3.  **Cleaning**.  Cleanse and preprocess data by deciding strategies to handle missing fields and alter the data as per the requirements.
4.  **Reduction and projection**.  Simplify the data sets by removing unwanted variables. Then, analyze useful features that can be used to represent the data, depending on the goal or task.
5.  **Method selection**.  Match KDD goals with data mining methods to suggest patterns.  What can each method tell you, and more importantly, what can it _not_ tell you?
6.  **Exploratory analysis**.  Choose  the  data-mining  algorithm(s)  and  selecting  method(s)to  be  used  for  searching  for  data  patterns
7.  **Data mining**.  Search for patterns of interest in a particular representational form, which include classification rules or trees, regression and clustering.
8.  **Interpretation**.  Interpret essential knowledge from the mined patterns.  _There be dragons here._
9.  **Action**.  Use the knowledge and incorporate it into another system for further action.
10.  **Reporting**.  Document it and make reports for interested parties.
(Altered from [[Fayyad96](https://www.kdnuggets.com/gpspubs/aimag-kdd-overview-1996-Fayyad.pdf)].)


##  Data Cleaning Example

-   [PI4 data cleaning lesson](https://github.com/pi4-uiuc/2017-bootcamp/blob/master/lessons/data/01-cleaning.md)


##  Data Classification Example

`scikit-learn` is a versatile package which provides a uniform interface for training and predicting from data sets.  Since Ram is going to cover clustering methods, presumably K-means clustering, I'll cover classification using the support vector machine method.

### 1.  Overview and Motivation

Consider two trees dropping seeds on the ground.  If one tree is an oak, it drops acorns around it in a normally-distributed pattern.  The other tree, a pecan tree, drops its nuts around it as well in a normally-distributed pattern.  If a new nut falls, we can use the pattern of dropped nuts to predict which it is more likely to be:  a pecan or an acorn.  We say that our data set has two _features_, or dimensions:  the $x$ and $y$ coordinates of the nut locations; and one _label_:  whether the nut is an acorn or a pecan.

If the trees are far apart, this classification becomes quite easy.  When the trees draw closer together, the problem becomes nontrivial and the error rate increases.  However, we can still predict with some accuracy which a nut is likely to be, based only on the features we observe about it:  its location on the ground.

In the simplest case, this is what a support vector machine can do for you:  recommend a classification based on known features.  However, SVMs are far more flexible than this.

First, plot the location of each fallen nut (and its label) on a rubber sheet.  At first, there is of course no difference between this rubber sheet and a regular sheet of paper, but if we start to tug and deform the sheet, we find that a clear division between the data values can be made.  Ideally, after pulling the sheet into a particular conformation, we would be able to draw a straight line between the data values to separate them.  If a new nut falls, we apply the same transformation to its position in order to make the separation (and thus the likely identity) clear.

Essentially, this is what support vector machines do for classification problems:  they find a transformation (expressed as a mathematical equation) which they can use to then cleanly separate two (or more) categories of data values.  This can take place in much higher dimensions that two as well:  any new feature of the data can be used to inform the support vector machine.

What are support vector machines good for—or rather, when would we choose to utilize SVMs instead of one of the many other methods available for classification?  Briefly, we want to use them when our data set is highly dimensional—in other words, when there are many things we know about our data (like age, sex, click-through rate, past purchasing habits, etc.).  SVMs are even effective when there are more features (or dimensions) to our data than the data samples we have available to train on.

Formally, we call the support vector machine a _supervised learning_ _kernel method_.  Supervised learning means that we need to train the SVM on a data set before we can use it to predict classifications or regressions subsequently.  Being a kernel method means that support vector machines use a mathematical trick (the kernel trick) internally to be computationally efficient in high-dimensional spaces.  The kernel provides a mathematical way of describing how "close" feature values—and thus data points—are to each other.

I will use the words "features" and "dimensions" interchangeably throughout this lesson.  These words contrast with "label" or "classification", also interchangeable.

-   [Kernel trick](https://en.wikipedia.org/wiki/Kernel_method)
-   [SVMs](https://infogalactic.com/info/Support_vector_machine)


### 2.  SVM for Classification

Let's implement the oak and pecan tree example now.  First, we need to set up a training data set.  If we stood outside and collected the data, we could create a real-world exercise.  Since we're not going to actually do that in this case, we will generate our data set using Python.  We need to describe two normal distributions, representing the fall of nuts from each tree.

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

[`numpy.random.normal`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html) accepts several parameters:

-   `loc` describes the center of the distribution, in this case the location of our tree;
-   `scale` describes the standard deviation of the distribution, in this case proportional to the width of the tree's branches; and
-   `size` describes the number of values we require, in this case the number of nuts which have fallen.

Let us suppose that an oak tree has branches spread wide, such that a standard deviation of 5 is appropriate.

First we describe our majestic oak as being located at $(-10,+10)$ in our field.  A thousand acorns have been plotted as having fallen from it:

    n_oak = 1000
    oak_locs = np.random.normal( loc=(-10,+10),scale=(5,5),size=(n_oak,2) )

    plt.plot( oak_locs[ :,0 ],oak_locs[ :,1 ],'ro' )
    plt.show()

Once we generate the data set, we will store it in a Pandas data frame for ease of access.  I will generally opt for clear Pandas statements, even if these are not as concise as the recommended way.

    df = pd.DataFrame( { 'Label' : pd.Series( 'oak',index=list( range( n_oak ) ) ),
                         'x' : oak_locs[ :,0 ],
                         'y' : oak_locs[ :,1 ] } )

The pecan tree will have a narrower span, let's say a standard deviation of 3.  Our pecan tree is located at coordinates $(+10,-10)$, and we have collected 500 acorns and plotted them as well:

    n_pec = 500
    pec_locs = np.random.normal( loc=(+10,-10),scale=(3,3),size=(n_pec,2) )

    df_pec = pd.DataFrame( { 'Label' : pd.Series( 'pecan',index=list( range( n_oak,n_oak+n_pec ) ) ),
                             'x' : pec_locs[ :,0 ],
                             'y' : pec_locs[ :,1 ] } )
    df = df.append( df_pec )

    axes = df[ df[ 'Label' ] == 'oak' ].plot( x='x',y='y',color='r',marker='o',linestyle='' )
    df[ df[ 'Label' ] == 'pecan' ].plot( x='x',y='y',color='b',marker='x',linestyle='',ax=axes )
    plt.xlim( ( -25,25 ) )
    plt.ylim( ( -25,25 ) )
    plt.show()

>  Since we are working with random numbers, your values will be slightly different from mine.  This should not give rise to any problems while completing this exercise.

From this figure, you can see that it would be reasonably straightforward for a human to draw a curve cleanly dividing the acorns from the pecans.  In this example, we will use the support vector machine method to identify that curve and use it to predictively classify newly discovered samples.

    from sklearn import svm

    clf = svm.SVC( kernel='linear' )
    clf.fit( df[ [ 'x','y' ] ],df[ 'Label' ] )

That's it—the convenience of the `scikit-learn` tools is that they have simple and consistent interfaces.  Now we will plot the underlying fields into which the algorithm divides acorns and pecans.

    X,Y = np.meshgrid( np.linspace( df[ 'x' ].min(),df[ 'x' ].max() ),np.linspace( df[ 'y' ].min(),df[ 'y' ].max() ) )
    X = X.ravel()
    X.shape = 2500,1
    Y = Y.ravel()
    Y.shape = 2500,1
    C = clf.predict( np.concatenate( ( X,Y ),axis=1 ) )

    fig,ax = plt.subplots()
    df_oak = df[ df[ 'Label' ] == 'oak' ]
    ax.plot( df_oak[ 'x' ],df_oak[ 'y' ],'ro',label='Acorns' )
    df_pec = df[ df[ 'Label' ] == 'pecan' ]
    ax.plot( df_pec[ 'x' ],df_pec[ 'y' ],'bx',label='Pecans' )

    for i in range( len( C ) ):
        if C[ i ] == 'oak':
            ax.plot( X[ i ],Y[ i ],'r+' )
        else:
            ax.plot( X[ i ],Y[ i ],'b+' )
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )
    plt.title( 'SVM Classification (Linear Kernel)' )
    plt.legend()
    plt.show()

To break this down, let's re-examine the initial code for the SVM classifier:

    clf = svm.SVC( kernel='linear' )

This line specifies that classification will occur using a linear kernel.  This means that only certain kinds of fits of our data, namely linear ones, are allowed—note that we observe a straight line dividing our acorns and pecans.

The next line,

    clf.fit( df[ [ 'x','y' ] ],df[ 'Label' ] )

means that we are using the $x$ and $y$ data to predict the data label.  Future predictions, then, mean that we pass an $(x,y)$ pair into the predictor:

    new_nut_xy = np.array( ( 1,1 ) )
    clf.predict( new_nut_xy.reshape( 1,-1 ) )

(Unfortunately, `scikit-learn` functions tend to be quite precise about the shape of the inputs, which means we have to liberally pepper our code with `concatenate` and `reshape` methods.)

Next, let's see what happens when we move the trees closer together.  At some point, we expect to be unable to separate the acorns from the pecans by a straight line.  This implies that we should be using other curves, such as polynomials and exponential functions, in the cases of more complicated data.

    # regenerate data, retrain, etc.

Recall the explanation involving the rubber plot.  By tugging and deforming the sheet (that is, transforming it by a function), we can attempt to find a clear division between the classes of data values.  In however many dimensions are necessary, we can draw a straight line, a plane, or a higher-dimensional analogue (the _hyperplane_) to separate the data.
