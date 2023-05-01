Download Link: https://assignmentchef.com/product/solved-b555-project-4-experiments-with-lda
<br>
In this programming project we implement Latent Dirichlet Allocation (LDA) and inspect its performance both in an unsupervised manner and when used as a preprocessing step for supervised learning. Your goals in this assignment are to (i) implement the collapsed Gibbs sampler for LDA inference, and (ii) compare the LDA topic representation to a “bag-of-words” representation with respect to how well they support document classification.

<h1>Data</h1>

Data for this assignment is provided in a zip file pp4data.zip on Canvas. Each dataset is given in a separate sub-directory.

We will use a subset of the well known <em>20 newsgroups </em>dataset. The subset consists of 200 documents that have been pre-processed and “cleaned” so <em>they do not require any further manipulation</em>, i.e., you only have to read the space-separated strings from the ASCII text files. Each document belongs to one of two classes. The file <em>index.csv </em>holds the true labels of each document. Labels are ignored in task 1 but used in task 2.

We have prepared an additional small dataset, <em>artificial</em>, for developing your implementation. Running your sampler on this dataset with <em>K </em>= 2 and parameters as below, you should find that the three most frequent words in the two topics are {<em>bank, river, water</em>} and {<em>dollars, bank, loan</em>} (not necessarily in those orders).

<h1>Task 1: Gibbs Sampling</h1>

In this portion, your task is to implement the collapsed Gibbs sampler for LDA. In the case of LDA, the output represents a sample of the (hidden) topic variables for each word. Recall that in LDA we sample the hidden topic variables associated with words in the text. This sample of topic variables can be used to calculate topic representations per document. Algorithm 1 describes one possible implementation of the collapsed Gibbs sampler.

<strong>Algorithm 1 </strong>Collapsed Gibbs sampler for LDA

<strong>Require: </strong>Number of topics <em>K</em>, Dirichlet parameter for topic distribution <em>α</em>, Dirichlet parameter for word distribution <em>β</em>, number of iterations to run sampler <em>N<sub>iters</sub></em>, array of word indices <em>w</em>(<em>n</em>), array of document indices <em>d</em>(<em>n</em>), and array of initial topic indices <em>z</em>(<em>n</em>), where <em>n </em>= 1<em>…N<sub>words </sub></em>and <em>N<sub>words </sub></em>is the total amount of words in the corpus.

1: Generate a random permutation <em>π</em>(<em>n</em>) of the set {1<em>,</em>2<em>,…,N<sub>words</sub></em>}

2: Initialize a <em>D</em>×<em>K </em>matrix of topic counts per document <em>C<sub>d</sub></em>, where <em>D </em>is the number of documents

3: Initialize a <em>K </em>× <em>V </em>matrix of word counts per topic <em>C<sub>t</sub></em>, where <em>V </em>is the number of words in the vocabulary

4: Initialize a 1 × <em>K </em>array of probabilities <em>P </em>(to zero)

5: <strong>for </strong><em>i </em>= 1 to <em>N<sub>iters </sub></em><strong>do </strong>6:              <strong>for </strong><em>n </em>= 1 to <em>N<sub>words </sub></em><strong>do</strong>

7:               <em>word </em>← <em>w</em>(<em>π</em>(<em>n</em>))

8:               <em>topic </em>← <em>z</em>(<em>π</em>(<em>n</em>))

9:               <em>doc </em>← <em>d</em>(<em>π</em>(<em>n</em>))

10:                <em>C<sub>d</sub></em>(<em>doc,topic</em>) ← <em>C<sub>d</sub></em>(<em>doc,topic</em>) − 1

11:                <em>C<sub>t</sub></em>(<em>topic,word</em>) ← <em>C<sub>t</sub></em>(<em>topic,word</em>) − 1

12:               <strong>for </strong><em>k </em>= 1 to <em>K </em><strong>do</strong>

<em>C k,word          β            C      doc,k</em>

13:

14:             <strong>end for</strong>

15:              <em>P </em>← normalize <em>P</em>

16:              <em>topic </em>← sample from <em>P</em>

<table>

 <tbody>

  <tr>

   <td width="95"></td>

  </tr>

  <tr>

   <td></td>

   <td></td>

  </tr>

 </tbody>

</table>

17:             <em>z</em>(<em>π</em>(<em>n</em>)) ← <em>topic</em>

18:                <em>C<sub>d</sub></em>(<em>doc,topic</em>) ← <em>C<sub>d</sub></em>(<em>doc,topic</em>) + 1

19:                <em>C<sub>t</sub></em>(<em>topic,word</em>) ← <em>C<sub>t</sub></em>(<em>topic,word</em>) + 1

20:        <strong>end for</strong>

21: <strong>end for</strong>

22: <strong>return </strong>{<em>z</em>(<em>n</em>)}<em>,C<sub>d</sub>,C<sub>t</sub></em>

For this project, fix the number of iterations to run the sampler at <em>N<sub>iters </sub></em>= 500. The Dirichlet parameter for the topic distribution is <em>α</em><strong>1 </strong>where <strong>1 </strong>is a vector of ones with <em>K </em>entries (<em>K </em>is the number of topics), and <em>α </em>= <em><sub>K</sub></em><u><sup>5 </sup></u>. The Dirichlet parameter for the word distribution is <em>β</em><strong>1 </strong>where <strong>1 </strong>is a vector of ones with <em>V </em>entries (<em>V </em>is the size of the vocabulary), and <em>β </em>= 0<em>.</em>01.

We suggest testing your implementation first on the <em>artificial </em>dataset with <em>K </em>= 2 because you know what to expect and the run time is shorter. Once you have verified that your implementation works correctly, run your sampler with <em>K </em>= 20 on the <em>20 newsgroups </em>dataset. After the sampler has finished running, output the 5 most frequent words of each topic into a CSV file, <em>topicwords.csv</em>, where each row represents a topic. Include these results in both your report and submission. In your report discuss the results obtained (i.e., the topics). Do the topics obtained make sense for the dataset?

Finally, you will need the topic representations for the next part. For a document <em>doc</em>, this will be a vector of <em>K </em>values, one for each topic, where the <em>k</em>th value is given by  and <em>C<sub>d </sub></em>is output from the sampler.

<h1>Task 2: Classification</h1>

In this portion we will evaluate the dimensionality reduction accomplished by LDA in its ability to support document classification and compare it to the bag of words representation.

The first step is to prepare the data files for the two representations. The first is given by the topic representation of the previous section, where each document is represented by a feature vector of length <em>K</em>. The second representation is the “bag-of-words” representation. This representation has a feature for each word in the vocabulary and the value of this feature is the number of occurrences of the corresponding word in the document.

For the evaluation we will reuse the logistic regression implementation from project 3 (that you developed as part of GLM). You should use the value <em>α </em>= 0<em>.</em>01 for the regularization parameter of logistic regression in this part.

Your task is to generate learning curves in the same way you did there: Step 1) Set aside 1/3 of the total data (randomly selected) to use as a test set. Step 2) Record performance as a function of increasing training set size (with each training set randomly selected from the other 2/3 of the total data). Repeat Steps 1 &amp; 2 a total of 30 times to generate learning curves with error bars (i.e., ±1<em>σ</em>). Performance is defined as classification accuracy on the test set.

Plot the learning curve performance of the logistic regression algorithm (with error bars) on the two representations. Then discuss your observations on the results obtained.

<h1>Additional Notes</h1>

<ul>

 <li>As in previous projects you may use I/O and math libraries (e.g., from numpy) but you should implement all machine learning portions of the algorithms yourself.</li>

 <li>Please submit the logistic regression implementation with this project so that your code can be used and tested without further manipulation.</li>

 <li>The run time for this project is non-negligible. A Matlab / Python implementation of the collapsed Gibbs sampler for the <em>20 newsgroups </em>dataset might take 10 or more minutes to run and might be longer if your implementation is not optimized. The many runs of logistic regression for the learning curves also require some time.</li>

</ul>