## Image processing

### Execution instructions 
First clone the git repository and then execute this command in your terminal: 
```shell
make
```

To execute the sobel filter run this command:
```shell
make test-sobel
```

To execute the sharpen filter run this command:
```shell
make test-sharpen
```

When the menu appears you must select path of the image or video in the options 1 and 3. The path is:
```shell
img/<file_name>
```

Finally, to remove the directories *obj* and *exec* run this command:
```shell
make clean
```

### Sobel filter example 
![Texto alternativo](/img/building.jpg) ![Texto alternativo](/img/building_sobel.png)

### Sharpen filter example
![Texto alternativo](/img/building.jpg) ![Texto alternativo](/img/building_sharpen.png)
