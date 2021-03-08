# Surface AI
This is an experimental AI library written natively in Rust.

Plans:
- [ ] Provide useful abstraction for generic use as a library ()
- [ ] Documentation
- [x] Perceptron
  - [x] Perceptron tests
  - [ ] Redesign perceptron/neuron type(s) such that code is more reusable in the neural net
- [x] Neural Network (still investigating is this is complete or not yet, see below)
  - [x] NN tests (created, but results are not satisfactory, see below)
- [ ] Allow for adjustable constants such as learning rate and training stop condition(s)
- [ ] Convolutional Neural Network
- [ ] Other machine learning algorithms

Currently I'm using this mostly as a learning experience, but I would like to extend it to be useful for other people in their machine learning projects. At the moment I am having difficulty using the NeuralNet type to classify stars on the [HR Diagram](https://www.kaggle.com/deepu1109/star-dataset) (as seen in the `test_star_classification` test). I am researching what's wrong, as I am still new to neural networks and how they function/how best to use them. Based on other tests (the sum learning example (`test_sum_classification`), and the test of a single back-propagation outcome (`test_learn_function`)) it appears as though the NN works correctly as intended, however I have noticed a case where the sum test does not terminate in training (or at least runs for a very long time). This is likely an issue with the NN, and (hopefully) the cause of the star classification not working correctly.

If you have advice or suggestions (or even contributions if you're interested) please either contact me through my email or by opening an issue/pull request on github.

I will only be releasing versions under "0.0.x" until I am happy enough with the library being presented as "usable" in a general sense. As mentioned, at the moment this is entirely experimental. The only reason I am releasing a crate this early is in the hopes that people with more experience will be able to play test/steer development of the project, and even also to allow the project to gain some traction and get used :D