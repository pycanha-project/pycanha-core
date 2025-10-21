# FAQ

I'll put here the problems I have encountered and how to solve them.

## CMake not found
An error you might get easily in Windows when trying to `conan install` or `conan create` the package: 

```
cmake : The term 'cmake' is not recognized
```

This is because `cmake` was not added to the **path**. You have three options:
1. Reinstall `cmake` and add it to the path when the installer ask for it
2. Add cmake to the `path` environment variable manually
3. Install a new cmake with pip (`pip install cmake`) that is detected in that python environment.

## Eigen3 not found after conan install
An error you might find after running `conan install` during the initial configuration and the build kits not being updated.

This happens because cmake tried to configure your project before conan was installed. to solve it:
1. Go to your project folder.
2. Manually delete the build folder (or the contents if the folder is locked).
3. Run `conan install` again.

This should fix the issue and you can now build the project correctly.