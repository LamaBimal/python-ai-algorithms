# Python AI Algorithms

Educational Python examples for graph search, regression, classification,
clustering, dimensionality reduction, and Nepali natural language processing.
The repository includes executable notebooks, reusable Python demos, datasets,
and regression tests.

## Examples

| Area | Algorithms and examples | Start here |
| --- | --- | --- |
| Graph search | BFS, DFS, uniform-cost search, greedy best-first search, and A* | [Search guide](SearchingTechnique/README.md) |
| Linear regression | Salary prediction from experience | [Salary notebook](LinearRegression/salary_prediction.ipynb) |
| Logistic regression | Diabetes classification and spam email detection | [Diabetes notebook](LogisticRegression/dibetics_prediction_classification.ipynb), [spam notebook](LogisticRegression/SpamEmailDetection.ipynb) |
| K-nearest neighbors | Diabetes classification | [KNN notebook](KNN/KNN_Classifier.ipynb) |
| Classification | Decision tree, random forest, SVM, and Gaussian Naive Bayes | [Machine-learning guide](MachineLearning/README.md) |
| Unsupervised learning | K-means clustering and PCA | [Additional algorithms notebook](MachineLearning/AdditionalAlgorithms.ipynb) |
| Nepali NLP | Text cleaning, suffix stemming, TF-IDF, logistic regression, and a feed-forward neural network | [NLP guide](NLP/ReadMe.md), [completed workshop](NLP/NLPWorkshop.ipynb) |

## Setup

Use Python 3.11 or newer and Git LFS. After installing Git LFS, clone the
repository and retrieve its tracked files:

```sh
git lfs install
git clone https://github.com/LamaBimal/python-ai-algorithms.git
cd python-ai-algorithms
git lfs pull
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install dependencies and launch Jupyter:

```sh
python -m pip install -r requirements.txt
python -m notebook
```

Select the virtual environment's Python kernel and run notebook cells in order.
Start Jupyter from the repository root or a directory inside it. Included
CSV files are found locally; Colab uploads and `/content` paths are unnecessary.
If a data file contains a short Git LFS pointer instead of records, run
`git lfs pull` before opening the notebook.

## Run Python demos

From the repository root with the environment activated:

```sh
python -m SearchingTechnique.breadth_first_search
python -m SearchingTechnique.deapth_first_search
python -m SearchingTechnique.uniform_cost_search
python -m SearchingTechnique.greedy_best_first_search
python -m MachineLearning.classification
python -m MachineLearning.clustering
python -m MachineLearning.dimensionality_reduction
```

The DFS filename retains its original spelling, `deapth_first_search.py`.
The [search case study](SearchingTechnique/Case_Study(BFS_&_A_).ipynb) demonstrates
BFS and A*. A* accepts unit-cost adjacency lists or dictionaries of neighbor
costs. Its coordinate example uses illustrative great-circle distances in
kilometers, rather than measured road distances. See the search guide for
input formats and shortest-path guarantees.

## Data and evaluation

- Salary and diabetes examples use the included CSV files. Diabetes imputation
  medians are learned after splitting, from training rows only.
- The spam notebook includes 40 synthetic teaching messages. Its accuracy is
  a pipeline demonstration, not a measure of real-world spam detection.
  To supply another CSV, set `MAIL_DATA_PATH` before launching Jupyter. It must
  contain `Category` (`ham` or `spam`) and `Message` columns.
- Both NLP notebooks use the included UTF-8 news and suffix files. They sample
  up to 200 articles per topic across sports, business, entertainment, and
  domestic news. TF-IDF replaces the earlier FastText download; no NLTK corpus,
  PyTorch installation, pretrained model, or GPU is needed. Reported scores
  describe the balanced teaching sample.
- Additional ML demos use scikit-learn's bundled datasets or generated data.
  They need no data downloads after dependency installation.

Text-model vocabularies and IDF weights are fitted only on training data.
The NLP models share a fixed stratified split. Use separate validation data
for tuning and reserve independent test data for final evaluation.

## Tests and CI

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The suite contains 34 tests covering search edge cases, dataset paths,
training-only preprocessing, model predictions, and every code cell in the
spam notebook and both NLP notebooks. Search checks use the standard library;
ML checks are skipped when their dependencies are missing.

To require all ML checks in PowerShell:

```powershell
$env:REQUIRE_ML_TESTS = '1'
python -m unittest discover -s tests -v
```

On macOS/Linux:

```sh
REQUIRE_ML_TESTS=1 python -m unittest discover -s tests -v
```

[GitHub Actions](.github/workflows/tests.yml) installs the development
requirements and runs the suite with ML checks required on Python 3.11 and
3.14 for pushes and pull requests.

## Roadmap

See [Future enhancements](FUTURE_ENHANCEMENTS.md) for the proposed release plan:
learning improvements and search visualizations, stronger evaluation, reusable
experiment tools, and broader algorithm coverage. The recommended next release
focuses on the learning experience. Version labels describe priorities, not
scheduled release dates.

## Background notes

The following notes introduce the AI concepts behind these examples.

### Intelligence

Intelligence is the ability to learn, understand, and use information to solve problems and adapt to new situations. It's not just about knowing lots of facts, but also being able to apply them in different ways. Intelligence can be measured by different tests, but it's important to remember that these tests only measure certain aspects of intelligence and aren't always accurate. Intelligence is influenced by both nature (genes) and nurture (environment and experiences), and it's something that can be improved with education and practice.

### Artificial Intelligence

Artificial Intelligence (AI) refers to computer programs and machines that can perform tasks that usually require human intelligence, like recognizing speech or images, making decisions, or even learning from data. AI systems use complex algorithms and mathematical models to analyze data and identify patterns, just like the way humans learn and make decisions.

AI is used in a wide range of applications today, from virtual assistants like Siri or Alexa to self-driving cars, online recommendation systems, fraud detection, and many more. AI is helping us solve problems and automate tasks that were previously impossible or very difficult to do.

### AI Perspective:
 - Acting and Thinking humanly
 - Acting and Thinking rationally

The **"acting and thinking humanly"** perspective is based on the idea that AI systems should be designed to mimic human thought processes and behaviors. This perspective is concerned with creating AI systems that can perceive the world like humans do, reason like humans do, and interact with humans in a way that is natural and intuitive. This approach often involves using techniques like natural language processing and computer vision to enable machines to understand and respond to human language and visual cues.

The **"acting and thinking rationally"** perspective, on the other hand, is concerned with creating AI systems that can make decisions and solve problems based on logical reasoning and formal rules. This perspective focuses on creating AI systems that can reason about complex systems and make decisions based on the best available evidence. This approach often involves using techniques like logic programming and decision theory to enable machines to reason and make decisions.

Both perspectives have their strengths and weaknesses. The "acting and thinking humanly" perspective is better suited for tasks that require natural language understanding, human-like creativity, and social intelligence. The "acting and thinking rationally" perspective, on the other hand, is better suited for tasks that require logical reasoning, problem-solving, and decision-making.

### History of AI

The history of Artificial Intelligence (AI) dates back to the mid-20th century, although the idea of creating machines that could think like humans goes back much further. Here is a brief timeline of the history of AI:

- 1950s: The term "Artificial Intelligence" was coined by John McCarthy, Marvin Minsky, and other computer scientists at Dartmouth College. The goal was to create machines that could reason, learn, and solve problems like humans.

- 1956: The Dartmouth Conference was held, which is considered to be the birthplace of AI. The conference brought together leading computer scientists to discuss AI research and lay the foundation for the field.

- 1960s-70s: AI research gained momentum, with advances in areas such as natural language processing, machine learning, and computer vision. Researchers developed algorithms for playing games like chess and checkers, and also explored the use of expert systems for decision-making.

- 1980s-90s: AI research hit a roadblock, as many of the initial promises of the field failed to materialize. Funding for AI research declined, and many researchers moved on to other areas of computer science.

- 2000s-present: Advances in computing power, data availability, and machine learning algorithms reignited interest in AI. Today, AI is being used in a wide range of applications, from self-driving cars to personalized medicine to virtual assistants.

Throughout its history, AI has faced a number of challenges, including the difficulty of representing knowledge and reasoning, the limitations of machine learning algorithms, and ethical concerns around the impact of AI on society. However, as technology continues to advance, AI is likely to play an increasingly important role in shaping the future of society.

### Foundations of AI :

**1. Philosophy:** Philosophy has contributed to the foundations of AI in many ways, including by developing theories of knowledge, reasoning, and language. For example, philosophers have developed theories of logic, probability, and decision theory, which are all essential to AI research.

**2. Economics:** Economics has contributed to the foundations of AI by providing a framework for understanding decision-making and resource allocation. Game theory, for example, is a branch of economics that studies strategic decision-making and is widely used in AI research.

**3. Psychology:** Psychology has contributed to the foundations of AI by providing insights into human cognition and perception. For example, research in cognitive psychology has helped develop algorithms for natural language processing and computer vision.

**4. Sociology:** Sociology has contributed to the foundations of AI by providing insights into how individuals and groups interact. Social network analysis, for example, is a technique used in AI research to analyze social networks and understand how information spreads.

**5. Linguistics:** Linguistics has contributed to the foundations of AI by providing insights into how language is used and understood. Natural language processing, for example, is a field of AI that is concerned with developing algorithms that can understand and generate human language.

**6. Neuroscience:** Neuroscience has contributed to AI research by providing insights into how the human brain processes information and controls behavior. For example, research in neuroscience has helped develop algorithms for machine learning and computer vision.

**7. Mathematics:** Mathematics is an essential foundation for AI research, providing tools for modeling and analyzing complex systems. Areas of mathematics that are particularly important for AI include calculus, linear algebra, probability theory, and optimization.

**8. Computer Science:** Computer Science is the primary field of study for AI research, providing the algorithms and computational tools for building intelligent systems. Areas of computer science that are particularly important for AI include machine learning, natural language processing, computer vision, and robotics.

**9. Control Theory:** Control theory is a branch of engineering that deals with the analysis and design of control systems. Control theory has contributed to AI research by providing tools for modeling and controlling complex systems, such as robots and autonomous vehicles.

### Applications of AI:
Artificial Intelligence (AI) is being used in a wide range of applications across different industries. Here are some examples of AI applications:

1. Healthcare: AI is being used in healthcare to improve diagnosis and treatment of diseases, as well as to improve patient outcomes. For example, machine learning algorithms are being used to analyze medical images and detect early signs of diseases such as cancer and Alzheimer's.

2. Transportation: AI is being used in transportation to improve safety and efficiency. Self-driving cars, for example, use AI algorithms to navigate roads and avoid obstacles.

3. Finance: AI is being used in finance to improve fraud detection, risk assessment, and trading. Machine learning algorithms are being used to analyze large amounts of financial data and make predictions about market trends.

4. Customer Service: AI is being used in customer service to improve customer satisfaction and reduce costs. Chatbots, for example, use natural language processing algorithms to interact with customers and provide support.

5. Manufacturing: AI is being used in manufacturing to improve efficiency and quality. Robots and autonomous machines, for example, use AI algorithms to perform tasks such as assembly and quality control.

6. Education: AI is being used in education to personalize learning and improve student outcomes. Adaptive learning platforms, for example, use AI algorithms to analyze student data and provide personalized learning experiences.

These are just a few examples of the many applications of AI. As technology continues to advance, AI is likely to play an increasingly important role in a wide range of industries and domains.
