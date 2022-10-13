//Imports
import { SharedStates as S } from "../support/sharedStates.js";
import * as App from "./app.js";
import * as C from "../support/constants.js";
import * as C_IPC from "../../../build/js/support/constants.js";
import * as Content from "./content.js";
import * as Index from "../index.js";

/**
 * @description The <strong>controls.js</strong> module contains properties and functions pertaining to the initialization and control of UI elements.
 * @requires app
 * @requires constants
 * @requires constantsIPC
 * @requires sharedStates
 * @module
 * 
 */
export {
    
    disableFrameViewControls,
    init,
    toggleLayout,
    updateExecuteButton,
    updateTheme
};

/**
 * @description An object containing the following members with module scope:
 * <ul>
 *      <li> Buttons </li>
 *      <li> ButtonsFrameView </li>
 *      <li> ScaleInterval </li>
 *      <li> ScaleTimeout </li>
 * </ul>
 * 
 * @private
 * @constant
 * 
 */
const M = {

    Buttons: undefined,
    ButtonsFrameView: undefined,
    ScaleInterval: undefined,
    ScaleTimeout: undefined
};

/**
 * @description Initializes the module.
 * @public
 * @function
 * 
 */
function init() {

    const execute = buildControlButton(
        
        C.HTMLElement.BUTTON_EXECUTE,
        C.ImageSource.RUN,
        C_IPC.Label.MENU_ITEM_RUN,
        App.executeCode
    );

    const settings = buildControlButton(
        
        C.HTMLElement.BUTTON_SETTINGS,
        C.ImageSource.SETTINGS,
        C_IPC.Label.MENU_ITEM_SETTINGS,
        App.displaySettingsDialog
    );

    const layoutHorizontal = buildControlButton(
        
        C.HTMLElement.BUTTON_LAYOUT_HORIZONTAL,
        C.ImageSource.DOCK,
        C_IPC.Label.MENU_ITEM_LAYOUT_HORIZONTAL,
        layoutButtonClickHandler
    );

    const layoutVertical = buildControlButton(
        
        C.HTMLElement.BUTTON_LAYOUT_VERTICAL,
        C.ImageSource.DOCK,
        C_IPC.Label.MENU_ITEM_LAYOUT_VERTICAL,
        layoutButtonClickHandler
    );

    const reset = buildControlButton(

        C.HTMLElement.BUTTON_RESET,
        C.ImageSource.RESET,
        C_IPC.Label.MENU_ITEM_RESET,
        Content.resetImageTransform
    );

    const reflectHorizontal = buildControlButton(

        C.HTMLElement.BUTTON_REFLECT_HORIZONTAL,
        C.ImageSource.REFLECT,
        C_IPC.Label.MENU_ITEM_REFLECT_HORIZONTALLY,
        reflectButtonClickHandler
    );

    const reflectVertical = buildControlButton(

        C.HTMLElement.BUTTON_REFLECT_VERTICAL,
        C.ImageSource.REFLECT,
        C_IPC.Label.MENU_ITEM_REFLECT_VERTICALLY,
        reflectButtonClickHandler
    );

    const scaleEvents = [C.Event.MOUSE_UP, C.Event.MOUSE_DOWN, C.Event.MOUSE_LEAVE];
    
    const scaleUp = buildControlButton(

        C.HTMLElement.BUTTON_SCALE_UP,
        C.ImageSource.SCALE_UP,
        C_IPC.Label.MENU_ITEM_SCALE_UP,
        scaleButtonClickHandler,
        scaleEvents
    );

    const scaleDown = buildControlButton(

        C.HTMLElement.BUTTON_SCALE_DOWN,
        C.ImageSource.SCALE_DOWN,
        C_IPC.Label.MENU_ITEM_SCALE_DOWN,
        scaleButtonClickHandler,
        scaleEvents
    );

    M.ButtonsFrameView = [

        reset,
        reflectHorizontal,
        reflectVertical,
        scaleUp,
        scaleDown
    ];

    M.Buttons = [
        
        execute,
        settings,
        layoutHorizontal,
        layoutVertical,
        ...M.ButtonsFrameView        
    ];

    updateExecuteButton();
    disableFrameViewControls();

    App.toggleLayout();
}

/**
 * @description Assigns attributes and event handling to an HTMLInputElement object that is of type "image".
 * @param {Object} button - The target HTMLInputElement object that is of type "image".
 * @param {string} src - The URL of the button's image.
 * @param {string} title - The label assigned as the button's tooltip.
 * @param {function} clickHandler - The callback function for the button's events.
 * @param {string[]} [events = [C.Event.CLICK]] - An array of event types assigned to the button.
 * @private
 * @function
 * 
 */
function buildControlButton(button, src, title, clickHandler, events = [C.Event.CLICK]) {

    button.src = src;
    button.title = title;
    button.classList.add(C.CSSClass.CONTROL_BUTTON);
    button.classList.add(S.Theme === C.Theme.LIGHT ? C.CSSClass.CONTROL_BUTTON_THEME_DARK : C.CSSClass.CONTROL_BUTTON_THEME_LIGHT);

    for (let event of events) {

        button.addEventListener(event, clickHandler);
    }

    return button;
}

/**
 * @description Event handler called when either of the layout control buttons are clicked.
 * @param {Object} event - The event object.
 * @private
 * @function
 * 
 */
function layoutButtonClickHandler(event) {

    const clickedVertical = (event.target === C.HTMLElement.BUTTON_LAYOUT_VERTICAL);
    S.Orientation = (clickedVertical) ? C_IPC.Orientation.VERTICAL : C_IPC.Orientation.HORIZONTAL;

    App.toggleLayout();
}

/**
 * @description Event handler called when either of the scale control buttons are clicked.
 * @param {Object} event - The event object.
 * @private
 * @function
 * 
 */
function scaleButtonClickHandler(event) {

    switch (event.type) {
            
        case C.Event.MOUSE_DOWN: {

            const scaleDirection = event.target === C.HTMLElement.BUTTON_SCALE_UP;
            const updateScale = () => Content.updateImageTransform(null, null, scaleDirection);
            
            updateScale();
            
            M.ScaleTimeout = setTimeout(() => M.ScaleInterval = setInterval(updateScale, 20), C.Measurement.SCALE_TIMEOUT);
            
            break;
        }
            
        case C.Event.MOUSE_UP:
        case C.Event.MOUSE_LEAVE:

            clearTimeout(M.ScaleTimeout);
            clearInterval(M.ScaleInterval);
            
            break;
    }
}

/**
 * @description Event handler called when either of the reflect control buttons are clicked.
 * @param {Object} event - The event object.
 * @private
 * @function
 * 
 */
function reflectButtonClickHandler(event) {
    
    const reflectH = (event.target === C.HTMLElement.BUTTON_REFLECT_HORIZONTAL) ? true : null;
    const reflectV = (event.target === C.HTMLElement.BUTTON_REFLECT_VERTICAL) ? true : null;
    
    Content.updateImageTransform(null, null, null, reflectH, reflectV);
}

/**
 * @description Determines whether or not to disable, enable, hide or show the Execute button.
 * @public
 * @function
 * 
 */
function updateExecuteButton() {

    const executeButton = C.HTMLElement.BUTTON_EXECUTE;
    const textArea = C.HTMLElement.TEXT_AREA;

    executeButton.disabled = (textArea.value.trim() === "" || S.AutoExecute);
    executeButton.style.display = (S.AutoExecute) ? C.CSS.NONE : C.CSS.BLOCK;

    Index.updateElectronRunMenuItem();
}

/**
 * @description Updates the theme for each control that has already been initialized.
 * @public
 * @function
 * 
 */
function updateTheme() {

    if (M.Buttons) {

        for (const button of M.Buttons) {

            button.classList.remove((S.Theme === C.Theme.DARK) ? C.CSSClass.CONTROL_BUTTON_THEME_DARK : C.CSSClass.CONTROL_BUTTON_THEME_LIGHT);
            button.classList.add((S.Theme === C.Theme.DARK) ? C.CSSClass.CONTROL_BUTTON_THEME_LIGHT : C.CSSClass.CONTROL_BUTTON_THEME_DARK);
        }
    }
}

/**
 * @description Alters the visual appearance of the layout control buttons according to the current orientation.
 * @public
 * @function
 * 
 */
function toggleLayout() {

    const isVertical = (S.Orientation === C_IPC.Orientation.VERTICAL);

    C.HTMLElement.BUTTON_LAYOUT_HORIZONTAL.disabled = (isVertical) ? false : true;
    C.HTMLElement.BUTTON_LAYOUT_VERTICAL.disabled   = (isVertical) ? true  : false;
}

/**
 * @description Disables or enables the Reset, Scale and Reflect buttons and info labels for the Frame View.
 * @public
 * @function
 * 
 */
function disableFrameViewControls(disabled = true) {

    if (M.ButtonsFrameView) {

        for (const button of M.ButtonsFrameView) {

            button.disabled = disabled;
        }

        const infoLabels = [

            C.HTMLElement.FRAME_VIEW_INFO_SCALE,
            C.HTMLElement.FRAME_VIEW_INFO_WIDTH,
            C.HTMLElement.FRAME_VIEW_INFO_HEIGHT
        ];

        for (let label of infoLabels) {

            label.textContent = "";
            label.classList.toggle(C.CSSClass.DISABLED_OPACITY);
        }
    }
}