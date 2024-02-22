# Playground

## About This Repository
This repository offers an opportunity to learn about the fundamental techniques of mathematical optimization and their implementation in Python. Each notebook includes explanations of mathematical optimization concepts along with practical code examples.

## How to Run Tutorials Using Binder
1. Click on the following Binder link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Jij-Inc/Playground/main)
2. Once the repository loads in Binder, a list of Jupyter notebooks within the `tutorials` directory will be displayed.
3. To start the tutorial, click on any notebook of your choice.
4. Execute the code cells in the notebook to experience the process of mathematical optimization.



## Configuration with TOML File
This repository uses a `config.toml` file for configuration. By default, the code looks for a `config.toml` file in the same directory. You need to update the `token` value in the `config.toml` with your personal token issued for the API.

Example `config.toml` for monthly use:
```toml
[default]
url = "https://api.m.jijzept.com/"
token = "your_personal_token"
```

Example `config.toml` for pay-as-you-go user:
```toml
[default]
url = "https://api.jijzept.com/"
token = "your_personal_token"
```

## QuickStart
1. Please access `quickstart`
2. Please Place `config.toml` in `quickstart directory`
3. Please run the notebook

## Notice
`tutorial` directory currently contains old-version ones. It will be updated shortly.


## Contact
For questions or feedback regarding this tutorial, please submit through GitHub Issues.
