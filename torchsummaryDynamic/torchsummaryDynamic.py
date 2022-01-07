from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

calc_op_types = (
    nn.Conv2d, nn.ConvTranspose2d,
    nn.Linear,
    nn.BatchNorm2d,
)

def summary(model, x, calc_op_types=calc_op_types, *args, **kwargs):
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """
    def register_hook(module):
        def hook(module, inputs, outputs):
            module_idx = len(summary)

            # Lookup name in a dict that includes parents
            for name, item in module_names.items():
                if item == module:
                    key = "{}_{}".format(module_idx, name)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = list(outputs[0].size())
                except AttributeError:
                    # pack_padded_seq and pad_packed_seq store feature into data attribute
                    info["out"] = list(outputs[0].data.size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params_nt"], info["params"], info["macs"] = 0, 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

                if name == "weight":
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate Mult-Adds in ConvNd
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                        assert len(inputs[0].size()) == 4 and len(inputs[0].size()) == len(outputs[0].size())+1

                        in_c, in_h, in_w = inputs[0].size()[1:]
                        k_h, k_w = module.kernel_size
                        out_c, out_h, out_w = outputs[0].size()
                        groups = module.groups
                        kernel_mul = k_h * k_w * (in_c // groups)

                        # conv
                        if isinstance(module, nn.Conv2d):
                            kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
                        # deconv
                        elif isinstance(module, nn.ConvTranspose2d):
                            # kernel_mul_group = kernel_mul * in_h * in_w * (out_c // groups)
                            kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)

                        total_mul = kernel_mul_group * groups
                        info["macs"] += total_mul
                    elif isinstance(module, nn.BatchNorm2d):
                        info["macs"] += inputs[0].size()[1]
                    else:
                        info["macs"] += param.nelement()

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        # ignore Sequential and ModuleList
        if isinstance(module, calc_op_types) or not module._modules:
            hooks.append(module.register_forward_hook(hook))

    module_names = get_names_dict(model)

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    try:
        with torch.no_grad():
            model(x) if not (kwargs or args) else model(x, *args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()

    # Use pandas to align the columns
    df = pd.DataFrame(summary).T

    df["Mult-Adds"] = pd.to_numeric(df["macs"], errors="coerce")
    df["Params"] = pd.to_numeric(df["params"], errors="coerce")
    df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
    df = df.rename(columns=dict(
        ksize="Kernel Shape",
        out="Output Shape",
    ))
    df_sum = df.sum()
    df.index.name = "Layer"

    df = df[["Kernel Shape", "Output Shape", "Params", "Mult-Adds"]]
    max_repr_width = max([len(row) for row in df.to_string().split("\n")])

    option = pd.option_context(
        "display.max_rows", 600,
        "display.max_columns", 10,
        "display.float_format", pd.io.formats.format.EngFormatter(use_eng_prefix=True)
    )
    with option:
        print("="*max_repr_width)
        print(df.replace(np.nan, "-"))
        print("-"*max_repr_width)
        df_total = pd.DataFrame(
            {"Total params": (df_sum["Params"] + df_sum["params_nt"]),
            "Trainable params": df_sum["Params"],
            "Non-trainable params": df_sum["params_nt"],
            "Mult-Adds": df_sum["Mult-Adds"]
            },
            index=['Totals']
        ).T
        print(df_total)
        print("="*max_repr_width)

    return df

def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                name = parent_name + "." + cls_name + "_"+ key if parent_name else key
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names
