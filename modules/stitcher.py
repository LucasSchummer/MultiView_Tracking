import cv2 as cv
from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.cropper import Cropper
from stitching.seam_finder import SeamFinder
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.blender import Blender


class Stitcher():

    def __init__(self, detector="orb", warper_type='spherical'):
        
        self.detector = detector
        self.cameras = None
        self.warper = Warper(warper_type=warper_type)
        self.cropper = Cropper()
        self.compensator = ExposureErrorCompensator()
        self.seam_masks = None
        self.ready = False

    def fit(self, img_paths):

        imgs = Images.of(list(img_paths))

        # Resize images to different resolutions
        medium_imgs = list(imgs.resize(Images.Resolution.MEDIUM))
        low_imgs = list(imgs.resize(Images.Resolution.LOW))

        # Detect features on medium images
        finder = FeatureDetector(detector=self.detector)
        features = [finder.detect_features(img) for img in medium_imgs]

        # Find matches
        matcher = FeatureMatcher(matcher_type='homography')
        matches = matcher.match_features(features)

        # Detect outliers / images impossible to align
        subsetter = Subsetter()
        indices = subsetter.get_indices_to_keep(features, matches)

        # Only estimate stitching parameters if possible to align all images
        if len(indices) < len(img_paths):
            print("Stitcher was unable to estimate the stitching parameters on the given set of images")
            return

        # Estimate camera parameters (intrinsic and extrinsic)) for each image
        camera_estimator = CameraEstimator()
        # Refine camera parameters using bundle adjustment 
        camera_adjuster = CameraAdjuster()
        # Wave correction
        wave_corrector = WaveCorrector()

        cameras = camera_estimator.estimate(features, matches)
        cameras = camera_adjuster.adjust(features, matches, cameras)
        self.cameras = wave_corrector.correct(cameras)

        # Setup warper (spherical by default)
        self.warper.set_scale(self.cameras)

        # Get low size and ratio medium/low resolution
        low_sizes = imgs.get_scaled_img_sizes(Images.Resolution.LOW)
        camera_aspect = imgs.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)
        # Warp low resolution images
        warped_low_imgs = list(self.warper.warp_images(low_imgs, cameras, camera_aspect))
        warped_low_masks = list(self.warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
        low_corners, low_sizes = self.warper.warp_rois(low_sizes, cameras, camera_aspect)

        # Get final size and ratio final/medium resolution
        final_sizes = imgs.get_scaled_img_sizes(Images.Resolution.FINAL)
        camera_aspect = imgs.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)
        # Warp final images
        warped_final_masks = list(self.warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
        final_corners, final_sizes = self.warper.warp_rois(final_sizes, cameras, camera_aspect)

        # Find the region to crop
        low_corners = self.cropper.get_zero_center_corners(low_corners)
        self.cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

        # Crop low resolution warped images
        cropped_low_masks = list(self.cropper.crop_images(warped_low_masks))
        cropped_low_imgs = list(self.cropper.crop_images(warped_low_imgs))
        low_corners, low_sizes = self.cropper.crop_rois(low_corners, low_sizes)

        # Crop final warped images based on the ROI found above
        lir_aspect = imgs.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL)
        cropped_final_masks = list(self.cropper.crop_images(warped_final_masks, lir_aspect))
        final_corners, final_sizes = self.cropper.crop_rois(final_corners, final_sizes, lir_aspect)

        # Find seam masks on low images
        seam_finder = SeamFinder()
        self.seam_masks = seam_finder.find(cropped_low_imgs, low_corners, cropped_low_masks)
        # Resize sea masks to finak resolution
        self.seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(self.seam_masks, cropped_final_masks)]

        # Evaluate the exposure correction to apply to each image
        self.compensator.feed(low_corners, cropped_low_imgs, cropped_low_masks)

        print("Stitching parameters have been successfully estimated")
        self.ready = True

        return self

    def stitch(self, img_paths):

        imgs = Images.of(list(img_paths))
        final_imgs = list(imgs.resize(Images.Resolution.FINAL))

        # Get final size and ratio final/medium since warping was estimated on medium images
        final_sizes = imgs.get_scaled_img_sizes(Images.Resolution.FINAL)
        camera_aspect = imgs.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)

        # Warp final images
        warped_final_imgs = list(self.warper.warp_images(final_imgs, self.cameras, camera_aspect))
        warped_final_masks = list(self.warper.create_and_warp_masks(final_sizes, self.cameras, camera_aspect))
        final_corners, final_sizes = self.warper.warp_rois(final_sizes, self.cameras, camera_aspect)

        # Get ratio final/low ratio since cropping ROI was obtained on low resolution images
        lir_aspect = imgs.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL)
        # Crop final images
        cropped_final_masks = list(self.cropper.crop_images(warped_final_masks, lir_aspect))
        cropped_final_imgs = list(self.cropper.crop_images(warped_final_imgs, lir_aspect))
        final_corners, final_sizes = self.cropper.crop_rois(final_corners, final_sizes, lir_aspect)

        # Apply exposure compensation
        compensated_imgs = [self.compensator.apply(idx, corner, img, mask) 
                        for idx, (img, mask, corner) 
                        in enumerate(zip(cropped_final_imgs, cropped_final_masks, final_corners))]
        
        # Blend images along seam masks (multiband blending)
        blender = Blender()
        blender.prepare(final_corners, final_sizes)
        for img, mask, corner in zip(compensated_imgs, self.seam_masks, final_corners):
            blender.feed(img, mask, corner)

        stitched, _ = blender.blend()

        return stitched

    def get_params(self):
        return (self.cameras, self.warper, self.cropper, self.compensator, self.seam_masks)

