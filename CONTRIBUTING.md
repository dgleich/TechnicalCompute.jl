# Contributing to TechnicalCompute.jl

Thank you for your interest in contributing to **TechnicalCompute.jl**! Your time and effort are greatly appreciated. This guide will help you understand how to contribute effectively to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Submitting Issues](#submitting-issues)
- [Contributing Code](#contributing-code)
- [Coding Standards](#coding-standards)
- [Thank You](#thank-you)

## Getting Started

Before you begin:

- **Discuss Your Ideas**: **TechnicalCompute.jl** is an opinionated package. Contributions that propose alternative approaches or opinions may not be accepted. It's highly recommended to **[open an issue](https://github.com/dgleich/TechnicalCompute.jl/issues)** to discuss your ideas before investing time in development.
- **Check Existing Issues and Pull Requests**: Someone might have already proposed your idea or reported the issue you're experiencing.

## Submitting Issues

When reporting bugs or requesting features:

- **Search Existing Issues**: To avoid duplicates, please check if your issue has already been reported.
- **Provide Detailed Information**:
  - **Expected Behavior**: Describe what you expected to happen when using the methods in isolation.
  - **Actual Behavior**: Explain what actually happened.
  - **Reproduction Steps**: Include specific code snippets or examples that demonstrate the issue.
  - **Environment Details**: Mention the versions of **TechnicalCompute.jl**, dependencies, and any other relevant system information.

### Reporting Package Interaction Issues

If you're experiencing issues with **TechnicalCompute.jl** not interacting properly the included packages:

- **Isolate the Issue**: Verify that the methods work as expected when not using `TechnicalCompute.jl`
- **Provide Code Examples**: Share specific code that fails when used with the `TechnicalCompute.jl` packages.
- **Check versions of packages**: Make sure you are comparing the same versions of the packages used in `TechnicalCompute.jl`
- **Describe Expected Behavior**: Explain how you expect the packages to interact.

## Contributing Code

We welcome code contributions that align with the project's goals. Focusing on demos of the awesome power of Julia using all of these tools as well as test cases are the most helpful! If there are areas of STEM that we need to include more modules on, let us know that too! 

## Coding Standards

- Follow Existing Code Style: Maintain consistency with the codebase. (2 spaces for tabs)
- Write Clear and Maintainable Code: Use meaningful variable and function names.
  We shouldn't have much code in TechnicalCompute.jl as it's supposed to be a meta package. 
  But some code may be unavoiadable. 
- Document Your Code: Include comments and documentation where necessary.

## Thank You

Your contributions help make TechnicalCompute.jl better for everyone. Thank you for taking the time to contribute!

**Largely written by ChatGPT and edited by David F. Gleich in response a prompt.**
> Can you help me write a GitHub contributing document?
> Some notes to include. 
> - This package needs to be opinionated. So contributions that discuss alternative opinions may not be accepted. It's better to post an issue to get a sense of things before working on something. 
> - Tests and demonstrations are highly encouraged. 
> - For filing bugs with packages not interacting, or a method not working properly. Report the expected behavior if the methods are used in isolation and a specific piece of code that fails.   