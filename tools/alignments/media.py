#!/usr/bin/env python3
""" Media items (Alignments, Faces, Frames)
    for alignments tool """

import logging
import os
import sys

import cv2
from tqdm import tqdm

# TODO imageio single frame seek seems slow. Look into this
# import imageio

from lib.align import Alignments, DetectedFace, update_legacy_png_header
from lib.image import (count_frames, generate_thumbnail, ImagesLoader,
                       png_write_meta, read_image, read_image_meta_batch)
from lib.utils import _image_extensions, _video_extensions, FaceswapError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AlignmentData(Alignments):
    """ Class to hold the alignment data """

    def __init__(self, alignments_file):
        logger.debug("Initializing %s: (alignments file: '%s')",
                     self.__class__.__name__, alignments_file)
        logger.info("[ALIGNMENT DATA]")  # Tidy up cli output
        folder, filename = self.check_file_exists(alignments_file)
        super().__init__(folder, filename=filename)
        logger.verbose("%s items loaded", self.frames_count)
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def check_file_exists(alignments_file):
        """ Check the alignments file exists"""
        folder, filename = os.path.split(alignments_file)
        if not os.path.isfile(alignments_file):
            logger.error("ERROR: alignments file not found at: '%s'", alignments_file)
            sys.exit(0)
        if folder:
            logger.verbose("Alignments file exists at '%s'", alignments_file)
        return folder, filename

    def save(self):
        """ Backup copy of old alignments and save new alignments """
        self.backup()
        super().save()

    def reload(self):
        """ Read the alignments data from the correct format """
        logger.debug("Re-loading alignments")
        self._data = self._load()
        logger.debug("Re-loaded alignments")

    def set_filename(self, filename):
        """ Set the :attr:`_file` to the given filename.

        Parameters
        ----------
        filename: str
            The full path and filename to set the alignments file name to
        """
        self._file = filename


class MediaLoader():
    """ Class to load images.

    Parameters
    ----------
    folder: str
        The folder of images or video file to load images from
    count: int or ``None``, optional
        If the total frame count is known it can be passed in here which will skip
        analyzing a video file. If the count is not passed in, it will be calculated.
    """
    def __init__(self, folder, count=None):
        logger.debug("Initializing %s: (folder: '%s')", self.__class__.__name__, folder)
        logger.info("[%s DATA]", self.__class__.__name__.upper())
        self._count = count
        self.folder = folder
        self.vid_reader = self.check_input_folder()
        self.file_list_sorted = self.sorted_items()
        self.items = self.load_items()
        logger.verbose("%s items loaded", self.count)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_video(self):
        """ Return whether source is a video or not """
        return self.vid_reader is not None

    @property
    def count(self):
        """ Number of faces or frames """
        if self._count is not None:
            return self._count
        if self.is_video:
            self._count = int(count_frames(self.folder))
        else:
            self._count = len(self.file_list_sorted)
        return self._count

    def check_input_folder(self):
        """ makes sure that the frames or faces folder exists
            If frames folder contains a video file return imageio reader object """
        err = None
        loadtype = self.__class__.__name__
        if not self.folder:
            err = f"ERROR: A {loadtype} folder must be specified"
        elif not os.path.exists(self.folder):
            err = f"ERROR: The {loadtype} location {self.folder} could not be found"
        if err:
            logger.error(err)
            sys.exit(0)

        if (loadtype == "Frames" and
                os.path.isfile(self.folder) and
                os.path.splitext(self.folder)[1].lower() in _video_extensions):
            logger.verbose("Video exists at: '%s'", self.folder)
            return cv2.VideoCapture(self.folder)
                # TODO ImageIO single frame seek seems slow. Look into this
                # retval = imageio.get_reader(self.folder, "ffmpeg")
        else:
            logger.verbose("Folder exists at '%s'", self.folder)
            return None

    @staticmethod
    def valid_extension(filename):
        """ Check whether passed in file has a valid extension """
        extension = os.path.splitext(filename)[1]
        retval = extension.lower() in _image_extensions
        logger.trace("Filename has valid extension: '%s': %s", filename, retval)
        return retval

    @staticmethod
    def sorted_items():
        """ Override for specific folder processing """
        return []

    @staticmethod
    def process_folder():
        """ Override for specific folder processing """
        return []

    @staticmethod
    def load_items():
        """ Override for specific item loading """
        return {}

    def load_image(self, filename):
        """ Load an image """
        if self.is_video:
            return self.load_video_frame(filename)
        src = os.path.join(self.folder, filename)
        logger.trace("Loading image: '%s'", src)
        return read_image(src, raise_error=True)

    def load_video_frame(self, filename):
        """ Load a requested frame from video """
        frame = os.path.splitext(filename)[0]
        logger.trace("Loading video frame: '%s'", frame)
        frame_no = int(frame[frame.rfind("_") + 1:]) - 1
        self.vid_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # pylint: disable=no-member
        _, image = self.vid_reader.read()
        # TODO imageio single frame seek seems slow. Look into this
        # self.vid_reader.set_image_index(frame_no)
        # image = self.vid_reader.get_next_data()[:, :, ::-1]
        return image

    def stream(self, skip_list=None):
        """ Load the images in :attr:`folder` in the order they are received from
        :class:`lib.image.ImagesLoader` in a background thread.

        Parameters
        ----------
        skip_list: list, optional
            A list of frame indices that should not be loaded. Pass ``None`` if all images should
            be loaded. Default: ``None``

        Yields
        ------
        str
            The filename of the image that is being returned
        numpy.ndarray
            The image that has been loaded from disk
        """
        loader = ImagesLoader(self.folder, queue_size=32, count=self._count)
        if skip_list is not None:
            loader.add_skip_list(skip_list)
        yield from loader.load()

    @staticmethod
    def save_image(output_folder, filename, image, metadata=None):
        """ Save an image """
        output_file = os.path.join(output_folder, filename)
        output_file = f"{os.path.splitext(output_file)[0]}.png"
        logger.trace("Saving image: '%s'", output_file)
        if metadata:
            encoded_image = cv2.imencode(".png", image)[1]
            encoded_image = png_write_meta(encoded_image.tobytes(), metadata)
            with open(output_file, "wb") as out_file:
                out_file.write(encoded_image)
        else:
            cv2.imwrite(output_file, image)  # pylint: disable=no-member


class Faces(MediaLoader):
    """ Object to load Extracted Faces from a folder.

    Parameters
    ----------
    folder: str
        The folder to load faces from
    alignments: :class:`lib.align.Alignments`, optional
        The alignments object that contains the faces. Used to update legacy hash based faces
        for <v2.1 alignments to png header based version. Pass in ``None`` to not update legacy
        faces (raises error instead). Default: ``None``
    """
    def __init__(self, folder, alignments=None):
        self._alignments = alignments
        super().__init__(folder)

    def process_folder(self):
        """ Iterate through the faces folder pulling out various information for each face.

        Yields
        ------
        dict
            A dictionary for each face found containing the keys returned from
            :class:`lib.image.read_image_meta_batch`
        """
        logger.info("Loading file list from %s", self.folder)

        if self._alignments is not None:  # Legacy updating
            filelist = [os.path.join(self.folder, face)
                        for face in os.listdir(self.folder)
                        if self.valid_extension(face)]
        else:
            filelist = [os.path.join(self.folder, face)
                        for face in os.listdir(self.folder)
                        if os.path.splitext(face)[-1] == ".png"]

        log_once = False
        for fullpath, metadata in tqdm(read_image_meta_batch(filelist),
                                       total=len(filelist),
                                       desc="Reading Face Data"):

            if "itxt" not in metadata or "source" not in metadata["itxt"]:
                if self._alignments is None:  # Can't update legacy
                    raise FaceswapError(
                        f"The folder '{self.folder}' contains images that do not include Faceswap "
                        "metadata.\nAll images in the provided folder should contain faces "
                        "generated from Faceswap's extraction process.\nPlease double check the "
                        "source and try again.")

                if not log_once:
                    logger.warning("Legacy faces discovered. These faces will be updated")
                    log_once = True
                if data := update_legacy_png_header(fullpath, self._alignments):
                    retval = data["source"]
                else:
                    raise FaceswapError(
                        f"Some of the faces being passed in from '{self.folder}' could not be matched to the alignments file '{self._alignments.file}'\nPlease double check your sources and try again."
                    )

            else:
                retval = metadata["itxt"]["source"]

            retval["current_filename"] = os.path.basename(fullpath)
            yield retval

    def load_items(self):
        """ Load the face names into dictionary.

        Returns
        -------
        dict
            The source filename as key with list of face indices for the frame as value
        """
        faces = {}
        for face in self.file_list_sorted:
            faces.setdefault(face["source_filename"], []).append(face["face_index"])
        logger.trace(faces)
        return faces

    def sorted_items(self):
        """ Return the items sorted by the saved file name.

        Returns
        --------
        list
            List of `dict` objects for each face found, sorted by the face's current filename
        """
        items = sorted(self.process_folder(), key=lambda x: (x["current_filename"]))
        logger.trace(items)
        return items


class Frames(MediaLoader):
    """ Object to hold the frames that are to be checked against """

    def process_folder(self):
        """ Iterate through the frames folder pulling the base filename """
        iterator = self.process_video if self.is_video else self.process_frames
        yield from iterator()

    def process_frames(self):
        """ Process exported Frames """
        logger.info("Loading file list from %s", self.folder)
        for frame in os.listdir(self.folder):
            if not self.valid_extension(frame):
                continue
            filename = os.path.splitext(frame)[0]
            file_extension = os.path.splitext(frame)[1]

            retval = {"frame_fullname": frame,
                      "frame_name": filename,
                      "frame_extension": file_extension}
            logger.trace(retval)
            yield retval

    def process_video(self):
        """Dummy in frames for video """
        logger.info("Loading video frames from %s", self.folder)
        vidname = os.path.splitext(os.path.basename(self.folder))[0]
        for i in range(self.count):
            idx = i + 1
            # Keep filename format for outputted face
            filename = "{}_{:06d}".format(vidname, idx)
            retval = {
                "frame_fullname": f"{filename}.png",
                "frame_name": filename,
                "frame_extension": ".png",
            }

            logger.trace(retval)
            yield retval

    def load_items(self):
        """ Load the frame info into dictionary """
        frames = {
            frame["frame_fullname"]: (
                frame["frame_name"],
                frame["frame_extension"],
            )
            for frame in self.file_list_sorted
        }

        logger.trace(frames)
        return frames

    def sorted_items(self):
        """ Return the items sorted by filename """
        items = sorted(self.process_folder(), key=lambda x: (x["frame_name"]))
        logger.trace(items)
        return items


class ExtractedFaces():
    """ Holds the extracted faces and matrix for
        alignments """
    def __init__(self, frames, alignments, size=512):
        logger.trace("Initializing %s: size: %s", self.__class__.__name__, size)
        self.size = size
        self.padding = int(size * 0.1875)
        self.alignments = alignments
        self.frames = frames
        self.current_frame = None
        self.faces = []
        logger.trace("Initialized %s", self.__class__.__name__)

    def get_faces(self, frame, image=None):
        """ Return faces and transformed landmarks
            for each face in a given frame with it's alignments"""
        logger.trace("Getting faces for frame: '%s'", frame)
        self.current_frame = None
        alignments = self.alignments.get_faces_in_frame(frame)
        logger.trace("Alignments for frame: (frame: '%s', alignments: %s)", frame, alignments)
        if not alignments:
            self.faces = []
            return
        image = self.frames.load_image(frame) if image is None else image
        self.faces = [self.extract_one_face(alignment, image) for alignment in alignments]
        self.current_frame = frame

    def extract_one_face(self, alignment, image):
        """ Extract one face from image """
        logger.trace("Extracting one face: (frame: '%s', alignment: %s)",
                     self.current_frame, alignment)
        face = DetectedFace()
        face.from_alignment(alignment, image=image)
        face.load_aligned(image, size=self.size, centering="head")
        face.thumbnail = generate_thumbnail(face.aligned.face, size=80, quality=60)
        return face

    def get_faces_in_frame(self, frame, update=False, image=None):
        """ Return the faces for the selected frame """
        logger.trace("frame: '%s', update: %s", frame, update)
        if self.current_frame != frame or update:
            self.get_faces(frame, image=image)
        return self.faces

    def get_roi_size_for_frame(self, frame):
        """ Return the size of the original extract box for
            the selected frame """
        logger.trace("frame: '%s'", frame)
        if self.current_frame != frame:
            self.get_faces(frame)
        sizes = []
        for face in self.faces:
            roi = face.aligned.original_roi.squeeze()
            top_left, top_right = roi[0], roi[3]
            len_x = top_right[0] - top_left[0]
            len_y = top_right[1] - top_left[1]
            if top_left[1] == top_right[1]:
                length = len_y
            else:
                length = int(((len_x ** 2) + (len_y ** 2)) ** 0.5)
            sizes.append(length)
        logger.trace("sizes: '%s'", sizes)
        return sizes
