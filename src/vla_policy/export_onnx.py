"""
VLA Model ONNX Export
"""
import torch
import onnx
from vla_policy.vla_network import VLAPolicy, VLAConfig


def export_to_onnx(checkpoint_path, output_path="models/vla_model.onnx", batch_size=1):
    config = VLAConfig()
    model = VLAPolicy(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy = (
        torch.randn(batch_size, 3, 224, 224),  # leader_rgb
        torch.randn(batch_size, 3, 224, 224),  # follower_rgb
        torch.randn(batch_size, 1, 224, 224),  # depth
        torch.randn(batch_size, 7),             # leader_joints
        torch.randn(batch_size, 6),             # follower_joints
        torch.randn(batch_size, 2),             # gripper_states
        torch.randn(batch_size, 512),           # lang_embed
        torch.randn(batch_size, 7),             # last_action
    )

    torch.onnx.export(model, dummy, output_path,
                      input_names=["leader_rgb", "follower_rgb", "depth",
                                   "leader_joints", "follower_joints",
                                   "gripper_states", "lang_embed", "last_action"],
                      output_names=["action", "value", "success_prob"],
                      dynamic_axes={name: {0: "batch_size"} for name in
                                    ["leader_rgb", "follower_rgb", "depth",
                                     "leader_joints", "follower_joints",
                                     "gripper_states", "lang_embed", "last_action",
                                     "action", "value", "success_prob"]},
                      opset_version=17)
    print(f"ONNX exported: {output_path}")

    model_check = onnx.load(output_path)
    onnx.checker.check_model(model_check)
    print("ONNX verification passed")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/vla/vla_model.onnx")
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output)
