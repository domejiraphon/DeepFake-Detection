def add_common_flags(parser):
    # model
    parser.add_argument(
        "--model_dir",
        type=str,
        default="e1",
        help="the path to store model directory in exp_folder",
    )
    parser.add_argument(
        "--exp_folder",
        type=str,
        default="runs",
        help="folder to store all experiment",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="restart experiment by deleting the old folder",
    )
    parser.add_argument(
        "--path2pretrained",
        type=str,
        default="./runs/pretrained/ours/sbi.tar",
        help="Pretrained of self blended network",
    )
    parser.add_argument(
        "--concat_segment",
        action="store_true",
        help="concatenate segmentation map to the input",
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="no validation log in terminal",
    )
    # dataset
    # TODO fix and double check the defaults everywhere.
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/scratch/jy3694/data/faceforensics2/original_sequences/youtube/raw/frames/",
        help="Path to vggface2 folder",
    )
    parser.add_argument(
        "--testset_dir",
        type=str,
        default="../assets/test_samples3",
        help="testset directory",
    )
    parser.add_argument(
        "--test_visualization_dir",
        type=str,
        default="e1",
        help="path to store grad cam images",
    )
    parser.add_argument(
        "--img_shape",
        type=int,
        default=224,
        help="Image size"
    )
    parser.add_argument(
        "--val_num_images",
        type=int,
        default=256,
        help="Number of validation images"
    )
    parser.add_argument(
        "--num_dataset",
        type=int,
        default=-1,
        help="Number of images used in training set"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Num worker for load dataset"
    )
    parser.add_argument(
        "--shuffle",
        default=True,
        action="store_true",
        help="shuffle dataset"
    )
    parser.add_argument(
        "--use_facenet",
        action="store_true",
        help="Use MTCNN in dataset to detect face",
    )
    parser.add_argument(
        "--pseudo_labels",
        action="store_true",
        help="Use pseudo_labels for regression head",
    )
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="Use lpips to guide regression head",
    )
    parser.add_argument(
        "--use_retinafacenet",
        action="store_true",
        help="Use retina facenet to detect face",
    )
    parser.add_argument(
        "--use_segment",
        action="store_true",
        help="Use segmentation map in dataset",
    )
    parser.add_argument(
        "--use_deeplab",
        action="store_true",
        help="Use deeplabv3+ for segmentation map in dataset",
    )
    parser.add_argument(
        "--deeplabv3_model",
        type=str,
        default="deeplabv3plus_mobilenet",
        help="type deeplabv3 model",
    )
    parser.add_argument(
        "--deeplabv3_ckpt",
        type=str,
        default="./runs/pretrained/deeplab_v3/latest_deeplabv3plus_mobilenet_face_occlude_os16.pth",
        help="Pretrained deeplabv3 ",
    )
    parser.add_argument(
        "--deeplabv3_num_classes",
        type=int,
        default=2,
        help="Number of classes of segmentation map"
    )
    parser.add_argument(
        "--output_stride",
        type=int,
        default=16,
        help="Output stride of segmentation map"
    )
    parser.add_argument(
        "--use_keypoint",
        action="store_true",
        help="Use keypoint in dataset",
    )

    parser.add_argument(
        "--use_self_blended",
        action="store_true",
        help="Use self blended technique to create a training set",
    )
    parser.add_argument(
        "--use_cutout",
        action="store_true",
        help="Use face cutout for the model",
    )
    parser.add_argument(
        "--use_occlusion",
        action="store_true",
        help="Use occluder for augmentation",
    )
    parser.add_argument(
        "--occluder_img_dataset",
        nargs="+",
        default=["/face_occlusion_generation/object_image_sr/",
                 "/face_occlusion_generation/11k_hands_img/"],
        help="Occlusion img path",
    )
    parser.add_argument(
        "--occluder_mask_dataset",
        nargs="+",
        default=["/face_occlusion_generation/object_mask_x4/",
                 "/face_occlusion_generation/11k_hands_masks/"],
        help="Occlusion mask path ",
    )
    parser.add_argument(
        "--segment_model",
        type=str,
        default='deeplabv3plus_mobilenet',
        choices=['deeplabv3plus_mobilenet', 'deeplabv3plus_resnet50',
                 'BiSeNet'],
        help="model for deeplabv3"
    )
    parser.add_argument(
        "--segment_ckpt",
        type=str,
        default='./checkpoints/latest_deeplabv3plus_mobilenet_face_occlude_os16',
        choices=[
            './checkpoints/latest_deeplabv3plus_mobilenet_face_occlude_os16',
            './checkpoints/latest_deeplabv3plus_resnet50_face_occlude_os16'],
        help="model for deeplabv3"
    )

    # visualzation
    parser.add_argument(
        "--visualized",
        default=False,
        action="store_true",
        help="To use neural network visualization technique for tensorboard",
    )
    parser.add_argument(
        "--visualization_type",
        choices=[
            "guided",
            "eigen",
            "gradcam",
            "scorecam",
            "gradcamplusplus",
            "ablationcam",
            "xgradcam",
            "eigencam",
            "fullgrad",
        ],
        default="eigencam",
        type=str,
        help="type of visualization technique",
    )

    # training
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set seed for repeatability"
    )
    parser.add_argument(
        "--lr", 
        default=1e-3, 
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--lr_decay", 
        default=0.1, 
        type=float,
        help="Learning rate decay"
    )
    parser.add_argument(
        "--num_epochs", 
        default=400, 
        type=int,
        help="Number of epochs"
    )
    parser.add_argument(
        "--print_header_every",
        type=int,
        default=100,
        help="Print the table header for how many epochs"
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print loss for how many steps"
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Log tensorboard for how many steps"
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=100,
        help="Log valset and testset tensorboard for how many steps"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="save ckpt for how many steps"
    )
    parser.add_argument(
        "--use_regression",
        action="store_true",
        help="use regression",
    )
    parser.add_argument(
            "--num_classes", type=int, default=2,
            help="num classes"
        )
    # test
    parser.add_argument(
        "--test",
        action="store_true",
        help="To test model on images",
    )
