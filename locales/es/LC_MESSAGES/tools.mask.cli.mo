��          �            x  +  y  E   �  �   �  �   n  7   -  i   e    �  �  �  �   �  �   T  �  �  =   �     �     �     �  ]  �  Y  >  ^   �  �   �  �   �  U   �  �   �  A  _  j  �  �     �   �    �  H   �      �      �      !                  	                                          
                   Apply gaussian blur to the mask output. Has the effect of smoothing the edges of the mask giving less of a hard edge. the size is in pixels. This value should be odd, if an even number is passed in then it will be rounded to the next odd number. NB: Only effects the output preview. Set to 0 for off Directory containing extracted faces, source frames, or a video file. Full path to the alignments file to add the mask to. NB: if the mask already exists in the alignments file it will be overwritten. Helps reduce 'blotchiness' on some masks by making light shades white and dark shades black. Higher values will impact more of the mask. NB: Only effects the output preview. Set to 0 for off Mask tool
Generate masks for existing alignments files. Optional output location. If provided, a preview of the masks created will be output in the given folder. R|How to format the output when processing is set to 'output'.
L|combined: The image contains the face/frame, face mask and masked face.
L|masked: Output the face/frame as rgba image with the face masked.
L|mask: Only output the mask as a single channel image. R|Masker to use.
L|bisenet-fp: Relatively lightweight NN based mask that provides more refined control over the area to be masked including full head masking (configurable in mask settings).
L|components: Mask designed to provide facial segmentation based on the positioning of landmark locations. A convex hull is constructed around the exterior of the landmarks to create a mask.
L|custom: A dummy mask that fills the mask area with all 1s or 0s (configurable in settings). This is only required if you intend to manually edit the custom masks yourself in the manual tool. This mask does not use the GPU.
L|extended: Mask designed to provide facial segmentation based on the positioning of landmark locations. A convex hull is constructed around the exterior of the landmarks and the mask is extended upwards onto the forehead.
L|vgg-clear: Mask designed to provide smart segmentation of mostly frontal faces clear of obstructions. Profile faces and obstructions may result in sub-par performance.
L|vgg-obstructed: Mask designed to provide smart segmentation of mostly frontal faces. The mask model has been specifically trained to recognize some facial obstructions (hands and eyeglasses). Profile faces may result in sub-par performance.
L|unet-dfl: Mask designed to provide smart segmentation of mostly frontal faces. The mask model has been trained by community members and will need testing for further description. Profile faces may result in sub-par performance. R|Whether the `input` is a folder of faces or a folder frames/video
L|faces: The input is a folder containing extracted faces.
L|frames: The input is a folder containing frames or is a video R|Whether to output the whole frame or only the face box when using output processing. Only has an effect when using frames as input. R|Whether to update all masks in the alignments files, only those faces that do not already have a mask of the given `mask type` or just to output the masks to the `output` location.
L|all: Update the mask for all faces in the alignments file.
L|missing: Create a mask for all faces in the alignments file where a mask does not previously exist.
L|output: Don't update the masks, just output them for review in the given output folder. This command lets you generate masks for existing alignments. data output process Project-Id-Version: faceswap.spanish
Report-Msgid-Bugs-To: 
PO-Revision-Date: 2022-08-05 14:05+0100
Last-Translator: 
Language-Team: tokafondo
Language: es_ES
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit
Plural-Forms: nplurals=2; plural=(n != 1);
Generated-By: pygettext.py 1.5
X-Generator: Poedit 3.0.1
 Aplica el desenfoque gaussiano a la salida de la máscara. Tiene el efecto de suavizar los bordes de la máscara dando menos de un borde duro. el tamaño está en píxeles. Este valor debe ser impar, si se pasa un número par se redondeará al siguiente número impar. NB: Sólo afecta a la vista previa de salida. Si se ajusta a 0, se desactiva Directorio que contiene las caras extraídas, los fotogramas de origen o un archivo de vídeo. Ruta completa del archivo de alineaciones al que se añadirá la máscara. Nota: si la máscara ya existe en el archivo de alineaciones, se sobrescribirá. Ayuda a reducir la "mancha" en algunas máscaras haciendo que los tonos claros sean blancos y los oscuros negros. Los valores más altos afectarán más a la máscara. NB: Sólo afecta a la vista previa de salida. Si se ajusta a 0, se desactiva Herramienta de máscara
Genera máscaras para los archivos de alineación existentes. Ubicación de salida opcional. Si se proporciona, se obtendrá una vista previa de las máscaras creadas en la carpeta indicada. R|Cómo formatear la salida cuando el procesamiento se establece en 'salida'.
L|combined: La imagen contiene la cara o fotograma, la máscara facial y la cara enmascarada.
L|masked: Da salida a la cara o fotograma como imagen rgba con la cara enmascarada.
L|mask: Sólo emite la máscara como una imagen de un solo canal. R|Máscara a utilizar.
L|bisenet-fp: Máscara relativamente ligera basada en NN que proporciona un control más refinado sobre el área a enmascarar, incluido el enmascaramiento completo de la cabeza (configurable en la configuración de la máscara).
L|components: Máscara diseñada para proporcionar una segmentación facial basada en la posición de los puntos de referencia. Se construye un casco convexo alrededor del exterior de los puntos de referencia para crear una máscara.
L|custom: Una máscara ficticia que llena el área de la máscara con 1 o 0 (configurable en la configuración). Esto solo es necesario si tiene la intención de editar manualmente las máscaras personalizadas usted mismo en la herramienta manual. Esta máscara no utiliza la GPU.
L|extended: Máscara diseñada para proporcionar una segmentación facial basada en el posicionamiento de las ubicaciones de los puntos de referencia. Se construye un casco convexo alrededor del exterior de los puntos de referencia y la máscara se extiende hacia arriba en la frente.
L|vgg-clear: Máscara diseñada para proporcionar una segmentación inteligente de rostros principalmente frontales y libres de obstrucciones. Los rostros de perfil y las obstrucciones pueden dar lugar a un rendimiento inferior.
L|vgg-obstructed: Máscara diseñada para proporcionar una segmentación inteligente de rostros principalmente frontales. El modelo de máscara ha sido entrenado específicamente para reconocer algunas obstrucciones faciales (manos y gafas). Los rostros de perfil pueden dar lugar a un rendimiento inferior.
L|unet-dfl: Máscara diseñada para proporcionar una segmentación inteligente de rostros principalmente frontales. El modelo de máscara ha sido entrenado por los miembros de la comunidad y necesitará ser probado para una mayor descripción. Los rostros de perfil pueden dar lugar a un rendimiento inferior. R|Si la entrada es una carpeta de caras o una carpeta frames o vídeo
L|faces: La entrada es una carpeta que contiene caras extraídas.
L|frames: La entrada es una carpeta que contiene fotogramas o es un vídeo R|Marcar esta opción dará como salida el fotograma completo, en vez de sólo el cuadro de la cara cuando se utiliza el procesamiento de salida. Sólo tiene efecto cuando se utilizan cuadros como entrada. R|Si se actualizan todas las máscaras en los archivos de alineación, sólo aquellas caras que no tienen ya una máscara del "tipo de máscara" dado o sólo se envían las máscaras a la ubicación "de salida".
L|all: Actualiza la máscara de todas las caras del archivo de alineación.
L|missing: Crea una máscara para todas las caras del fichero de alineaciones en las que no existe una máscara previamente.
L|output: No actualiza las máscaras, sólo las emite para su revisión en la carpeta de salida dada. Este comando permite generar máscaras para las alineaciones existentes. datos salida proceso 