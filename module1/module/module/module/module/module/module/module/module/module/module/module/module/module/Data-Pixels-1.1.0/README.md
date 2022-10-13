<div align="center">

<img src="./resources/source/images/icons/icon.png" width="500">   
<h1><strong>DATA PIXELS ／ </strong>Create Pixel Art Programmatically</h1>

[![npm](https://img.shields.io/npm/v/data-pixels.svg?style=flat&label=NPM%20Package)](https://www.npmjs.com/package/data-pixels)
[![npm](https://img.shields.io/npm/dt/data-pixels.svg?style=flat&label=NPM%20▼)](https://www.npmjs.com/package/data-pixels)
[![Github All Releases](https://img.shields.io/github/downloads/gmattie/Data-Pixels/total.svg?style=flat&label=GitHub%20▼)](https://github.com/gmattie/Data-Pixels/releases)
[![GitHub stars](https://img.shields.io/github/stars/gmattie/Data-Pixels.svg?style=flat&label=GitHub%20★)](https://github.com/gmattie/Data-Pixels/stargazers)
[![license](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flat)](https://github.com/gmattie/Data-Pixels/blob/master/resources/build/license)

</div>

The **DataPixels.js** source code facilitates production of pixel art that is entirely generated programmatically at runtime.   Additionally, the accompanying desktop application, **Data Pixels Playground**, may be used to write and execute code for displaying both customized and automated pixel art.

## **DataPixels.js** 

The DataPixels.js source code features a modular, ES2015 Class design for accessible and effortless construction of new instances.  Each instance contains both **HTMLCanvasElement** and **HTMLImageElement** public accessors whose sources consist of the programmatically generated pixel art.

#### **Install**

DataPixels.js package can be installed as a project dependency from NPM by entering the following CLI command:

```
npm i data-pixels
```

#### **Create**

The DataPixels.js constructor requires 2 arguments:

1. **pixelData**:  An array containing one or more arrays of equal length, consisting of strings composed of 0-255 integer values per 24-bit RGB color channel (e.g., `“255, 255, 255”`) or 32-bit RGBA color channel (e.g., `“255, 255, 255, 255”`).  Additionally, the strings may optionally contain any kind of descriptive text (e.g., `“Red: 255, G - 128, 64 for Blue, Transparency = 32”`) as only the number values within the string will be parsed in RGB / RGBA order.  Strings that contain more than 4 numbers will throw an error.

2. **pixelSize**:  The size of each color data unit in pixels.  This value represents the size of each perceived pixel that forms the pixel art.

For more information see the [**Example Code**](https://github.com/gmattie/Data-Pixels#example-code) section below.

![Code Output](./resources/source/images/readme/CodeMario.png)

## **Data Pixels Playground**

Data Pixels Playground is a lightweight, cross-platform, desktop application for **Windows**, **Mac** and **Linux**, which may be used to write and execute DataPixels.js instances for previewing and testing purposes.

The application features **built-in example code** via the *Help* menu as well as the ability to **parse pixel data from image files** to produce automatically generated code through the *File > Open Image File…* menu item or through drag-and-drop gestures.  Additionally, compiled Data Pixels visible in the *View Panel* can be **easily dragged, scaled and reflected** using the mouse or in-app controls to help you design the perfect images for your requirements.

Note: pixel color values that are automatically interpreted from image files with an embedded color space may differ slightly from the image’s intended color values.

![Application Screenshot](./resources/source/images/readme/ApplicationScreenshot.png)

## **Desktop Application Release Builds**
Creating release builds for **Windows**, **Mac** and/or **Linux** is a 2-step process: code compilation, then application packaging, both of which are accomplished by running command-line NPM scripts.

#### **Compilation**

Production code compilation can be executed by entering the following CLI command at the project **root folder** [*~/DataPixels/* ]:

```
npm run prod
```

For more detailed information concerning code compilation please refer to [**Project Foundation**](https://github.com/gmattie/Project-Foundation).

#### **Packaging**

Application packaging can be executed for individual platforms by entering one of the following CLI commands at the project **build folder** [*~/DataPixels/resources/build/* ]:

```
npm run package-linux
```

```
npm run package-mac
```

```
npm run package-windows
```

Note: In order to avoid problems with code signing and other build issues it is highly recommended to execute packaging scripts for an individual platform from its own operating system.

For more detailed information concerning application packaging please refer to [**Electron Packager**](https://github.com/electron-userland/electron-packager).

## **Example Code**

#### **Basic**

```javascript
/**
 * @description DataPixels basic code example
 * 
 */
import DataPixels from "data-pixels";

const R = "255, 0, 0, 255";    //Red
const G = "0, 255, 0, 255";    //Green
const B = "0, 0, 255, 255";    //Blue
const _ = "0, 0, 0, 0";        //Transparent

const data = [[R, G],
              [B, _]];

const size = 100;

const image = new DataPixels(data, size).image;

document.body.appendChild(image);
```

#### **Hearts**

```javascript
/**
 * @description DataPixels example of multiple heart shaped instances of different hues and rotations
 * 
 */
import DataPixels from "data-pixels";

/**
 * @description Properties of type <strong>{number}</strong> consist of:
 * <ul>
 *     <li> S </li>
 *     <li> M </li>
 *     <li> L </li>
 *     <li> XL </li>
 * </ul>
 * 
 * @constant
 * 
 */
const Size = {

    S: 10,
    M: 15,
    L: 20,
    XL: 25
};

/**
 * @description Creates a new heart shaped pixelData object of a specified hue
 * @param {number} red - The hue's red value 
 * @param {number} green - The hue's green value
 * @param {number} blue - The hue's blue value
 * @param {number} lightness - The hue's applied value for brightness and darkness
 * 
 */
function createPixelDataHeart(red, green, blue, lightness = 20) {

    const R = red;
    const G = green;
    const B = blue;
    const L1 = lightness;
    const L2 = L1 * 2;
    const H = `${R}, ${G}, ${B}`;                  //Main Hue
    const A = `${R + L2}, ${G + L2}, ${B + L2}`;   //Main Hue Light
    const Y = `${R - L1}, ${G - L1}, ${B - L1}`;   //Main Hue Dark
    const Z = `${R - L2}, ${G - L2}, ${B - L2}`;   //Main Hue Darker
    const $ = "0, 0, 0";                           //Stroke
    const _ = "0, 0, 0, 0";                        //Transparent

    return [[_, _, $, $, $, _, _, _, $, $, $, _, _],
            [_, $, H, H, H, $, _, $, Y, Y, Z, $, _],
            [$, H, H, A, H, H, $, H, H, Y, Y, Z, $],
            [$, H, A, H, H, H, H, H, H, H, Y, Z, $],
            [$, H, A, H, H, H, H, H, H, H, Y, Z, $],
            [$, H, H, H, H, H, H, H, H, H, Y, Z, $],
            [_, $, H, H, H, H, H, H, H, Y, Z, $, _],
            [_, _, $, H, H, H, H, H, H, Y, $, _, _],
            [_, _, _, $, H, H, H, H, Y, $, _, _, _],
            [_, _, _, _, $, H, H, Y, $, _, _, _, _],
            [_, _, _, _, _, $, Y, $, _, _, _, _, _],
            [_, _, _, _, _, _, $, _, _, _, _, _, _]];
}

/* 
 * Create multiple heart canvases of different colors and sizes
 *
 */ 
const redHeart = new DataPixels(createPixelDataHeart(200, 0, 50), Size.XL).canvas;
const blueHeart = new DataPixels(createPixelDataHeart(50, 30, 210), Size.L).canvas;
const purpleHeart = new DataPixels(createPixelDataHeart(125, 70, 180), Size.M).canvas;
const greenHeart = new DataPixels(createPixelDataHeart(25, 160, 50), Size.S).canvas;

/* 
 * Create a container for the heart canvases
 *
 */ 
const heartsContainer = document.createElement("div");
heartsContainer.style.filter = "drop-shadow(0 0 30px #FFFFFF)";

document.body.appendChild(heartsContainer);

/* 
 * Rotate and append each heart canvas to the container
 *
 */ 
const rotationDelta = 15;

[redHeart, blueHeart, purpleHeart, greenHeart].forEach((heart, index) => {

    heart.style.transform = `rotate(${rotationDelta * index}deg)`;
    heartsContainer.appendChild(heart);
});
```

#### **Mario Bros**

```javascript
/**
 * @description DataPixels example of Nintendo's Mario Brothers
 * 
 */
import DataPixels from "data-pixels";

/**
 * @description Creates a Mario or Luigi pixelData object
 * @param {boolean} isMario - Defines the brother's hat and shirt color.  Mario is red and Luigi is green. 
 * 
 */
function createMarioBrother(isMario = true) {

    const mainColor = (isMario) ? "255, 0, 0" : "0, 180, 0";

    const C = mainColor;        //Hat & Shirt
    const B = "100, 50, 0";     //Brown Hair & Boots
    const S = "255, 200, 150";  //Skin Tone
    const O = "0, 0, 255";      //Blue Overalls
    const Y = "255, 255, 0";    //Yellow Buckles       
    const W = "255, 255, 255";  //White Gloves
    const _ = "0, 0, 0, 0";     //Transparent (RGBA Format)

    return [[_, _, _, C, C, C, C, C, _, _, _, _],
            [_, _, C, C, C, C, C, C, C, C, C, _],
            [_, _, B, B, B, S, S, B, S, _, _, _], 
            [_, B, S, B, S, S, S, B, S, S, S, _],
            [_, B, S, B, B, S, S, S, B, S, S, B],
            [_, B, B, S, S, S, S, B, B, B, B, _],
            [_, _, _, S, S, S, S, S, S, S, _, _],
            [_, _, C, C, O, C, C, C, C, _, _, _],
            [_, C, C, C, O, C, C, O, C, C, C, _],
            [C, C, C, C, O, O, O, O, C, C, C, C],
            [W, W, C, O, Y, O, O, Y, O, C, W, W],
            [W, W, W, O, O, O, O, O, O, W, W, W],
            [W, W, O, O, O, O, O, O, O, O, W, W],
            [_, _, O, O, O, _, _, O, O, O, _, _],
            [_, B, B, B, _, _, _, _, B, B, B, _],
            [B, B, B, B, _, _, _, _, B, B, B, B]];
}

/*
 * Create and append a Mario Brother canvas instance
 * 
 */
const pixelSize = 30;

const brother = new DataPixels(createMarioBrother(true), pixelSize).canvas;
brother.style.filter = "drop-shadow(0 10px 20px #000000)";

document.body.appendChild(brother);
```

## **License**

[**MIT License**](./resources/build/license)

Copyright © 2017-2018 Geoffrey Mattie

