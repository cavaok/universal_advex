import torch
import pickle
from torch import optim
from torch import nn
import torch.nn.functional as F
from helper import create_diffuse_one_hot, set_DoC
from supabase_logger import SupabaseLogger
from data import get_mnist_digit_loaders
from auto import load_autoencoder64_model, load_autoencoder128_model, load_autoencoder256_model, load_autoencoder512_model, load_funky_autoencoder_model
from mlp import load_mlp_model
from hadamard import load_hadamard_model


def load_all_models():
    models = {}

    # Load MLP model
    models['mlp'] = load_mlp_model("models_backup/mlp_elu_512_256_128.pth")

    # Load base autoencoder64 model (single iteration version)
    models['auto64_1'] = load_autoencoder64_model(
        "models_backup/encoder_auto_64_elu_1.pth",
        "models_backup/decoder_auto_64_elu_1.pth"
    )

    # Load autoencoder64 sum models (2-4 iterations)
    for x in range(2, 5):
        models[f'auto64_{x}'] = load_autoencoder64_model(
            f"models_backup/encoder_auto_64_elu_{x}_sum.pth",
            f"models_backup/decoder_auto_64_elu_{x}_sum.pth"
        )

    # Load base autoencoder128 model (single iteration version)
    models['auto128_1'] = load_autoencoder128_model(
        "models_backup/encoder_auto_128_elu_1.pth",
        "models_backup/decoder_auto_128_elu_1.pth"
    )

    # Load autoencoder128 sum models (2-7 iterations)
    for x in range(2, 8):
        models[f'auto128_{x}'] = load_autoencoder128_model(
            f"models_backup/encoder_auto_128_elu_{x}_sum.pth",
            f"models_backup/decoder_auto_128_elu_{x}_sum.pth"
        )

    # Load base autoencoder256 model (single iteration version)
    models['auto256_1'] = load_autoencoder256_model(
        "models_backup/encoder_auto_256_elu_1.pth",
        "models_backup/decoder_auto_256_elu_1.pth"
    )

    # Load autoencoder256 sum models (2-10 iterations)
    for x in range(2, 11):
        models[f'auto256_{x}'] = load_autoencoder256_model(
            f"models_backup/encoder_auto_256_elu_{x}_sum.pth",
            f"models_backup/decoder_auto_256_elu_{x}_sum.pth"
        )

    # Load base autoencoder512 model (single iteration version)
    models['auto512_1'] = load_autoencoder512_model(
        "models_backup/encoder_auto_512_elu_1.pth",
        "models_backup/decoder_auto_512_elu_1.pth"
    )

    # Load autoencoder512 sum models (2-10 iterations)
    for x in range(2, 11):
        models[f'auto512_{x}'] = load_autoencoder512_model(
            f"models_backup/encoder_auto_512_elu_{x}_sum.pth",
            f"models_backup/decoder_auto_512_elu_{x}_sum.pth"
        )

    # Load funky autoencoder models (1-10 iterations)
    for x in range(1, 11):
        models[f'funkyauto_{x}'] = load_funky_autoencoder_model(
            f"models_backup/model_794_elu_{x}.pth"
        )

    # Load hadamard models (1-3 iterations)
    for x in range(1, 4):
        models[f'hadamard_{x}'] = load_hadamard_model(
            f"models_backup/f1_hadamard_{x}.pth",
            f"models_backup/f2_hadamard_{x}.pth"
        )

    return models


def load_adversarial_cases():
    # Load pickle data
    with open('doc.pkl', 'rb') as f:
        data = pickle.load(f)

    # Get digit loaders
    digit_loaders = get_mnist_digit_loaders(batch_size=1)

    cases_per_image = 18
    images_per_digit = 5
    all_cases = []

    for digit in range(10):
        digit_iterator = iter(digit_loaders[digit])

        for image_idx in range(images_per_digit):
            image, label = next(digit_iterator)

            start_idx = (digit * images_per_digit + image_idx) * cases_per_image
            end_idx = start_idx + cases_per_image
            current_cases = data[start_idx:end_idx]

            for case in current_cases:
                all_cases.append((image, label, case[0], case[1]))

    return all_cases


def mlp_advex_train(model, image, label, target, device, lambda_=0.5, num_steps=300, lr=0.01):
    # Setup
    image = image.to(device).view(-1, 784)  # Reshape for MLP
    target = target.to(device)
    mlp_image = image.clone().detach().requires_grad_(True)
    original_image = image.clone().detach()

    # Get original prediction for logging
    with torch.no_grad():
        mlp_prediction = model(mlp_image)
        mlp_prediction_label = F.softmax(mlp_prediction, dim=1)

    # Optimization setup
    optimizer = optim.Adam([mlp_image], lr=lr)

    # Training loop
    for step in range(num_steps):
        output = model(mlp_image)
        probs = F.softmax(output, dim=1)

        mlp_label_loss = F.kl_div(probs.log(), target)
        mlp_image_loss = F.mse_loss(mlp_image, original_image)
        mlp_loss = mlp_image_loss + lambda_ * mlp_label_loss

        optimizer.zero_grad()
        mlp_loss.backward()

        if step % 50 == 0:
            print(f"\nMLP Step {step + 1}/{num_steps}:")
            print(f"  Current probs: {probs.detach().cpu().numpy().round(3)}")
            print(f"  Target probs: {target.cpu().numpy().round(3)}")
            print(f"  Label Loss: {mlp_label_loss.item():.4f}")
            print(f"  Image Loss: {mlp_image_loss.item():.4f}")
            print(f"  Total Loss: {mlp_loss.item():.4f}")
            print(f"  Image grad max: {mlp_image.grad.abs().max().item()}")

        optimizer.step()

        with torch.no_grad():
            mlp_image.data.clamp_(0, 1)

    # Final evaluation
    with torch.no_grad():
        final_prediction = model(mlp_image)
        final_probs = F.softmax(final_prediction, dim=1)

        # Calculate metrics
        label_divergence = F.kl_div(final_probs.log(), target, reduction='sum')
        mse = F.mse_loss(mlp_image.view(1, -1), original_image.view(1, -1))
        frob = torch.norm(mlp_image.view(1, -1) - original_image.view(1, -1), p='fro')

    # Return results
    return {
        "adversarial_image": mlp_image.clone().detach(),
        "prediction": final_probs.clone().detach(),
        "original_prediction": mlp_prediction_label.clone().detach(),
        "label_kld": label_divergence.item(),
        "mse": mse.item(),
        "frob": frob.item()
    }


def auto_hadamard_advex_train(model, image, label, target, device, lambda_=1.0, num_steps=300, lr=0.01):
    # Setup
    image_dim = 28 * 28
    num_classes = 10

    # Prepare inputs - flatten image to 2D
    image = image.to(device).view(1, -1)  # Reshape to [batch_size, flattened_image]
    target = target.to(device)
    image_part = image.clone().detach().requires_grad_(True)
    label_part = torch.zeros(1, num_classes, device=device)
    label_part[0, label.item()] = 1  # Create one-hot encoding

    # Store original image for loss calculation
    with torch.no_grad():
        concat_input = torch.cat((image, label_part), dim=1)
        original_output = model(concat_input)
        original_image = original_output[:, :image_dim].clone().detach()
        original_prediction = F.softmax(original_output[:, image_dim:], dim=1)

    # Optimization setup
    optimizer = optim.Adam([image_part], lr=lr)

    # Training loop
    for step in range(num_steps):
        current_input = torch.cat((image_part, label_part), dim=1)
        output = model(current_input)

        # Split output into image and label parts
        output_label = output[:, image_dim:]
        output_probs = F.softmax(output_label, dim=1)

        # Calculate losses
        label_loss = F.kl_div(output_probs.log(), target)
        image_loss = F.mse_loss(image_part, original_image)
        total_loss = image_loss + lambda_ * label_loss

        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()

        if step % 50 == 0:
            print(f"\nStep {step + 1}/{num_steps}:")
            print(f"  Current probs: {output_probs.detach().cpu().numpy().round(3)}")
            print(f"  Target probs: {target.cpu().numpy().round(3)}")
            print(f"  Label Loss: {label_loss.item():.4f}")
            print(f"  Image Loss: {image_loss.item():.4f}")
            print(f"  Total Loss: {total_loss.item():.4f}")
            print(f"  Image grad max: {image_part.grad.abs().max().item()}")

        optimizer.step()

        # Clamp values to valid range
        with torch.no_grad():
            image_part.data.clamp_(0, 1)

    # Final evaluation
    with torch.no_grad():
        final_input = torch.cat((image_part, label_part), dim=1)
        final_output = model(final_input)
        final_probs = F.softmax(final_output[:, image_dim:], dim=1)

        # Calculate metrics
        label_divergence = F.kl_div(final_probs.log(), target, reduction='sum')
        mse = F.mse_loss(image_part.view(1, -1), original_image.view(1, -1))
        frob = torch.norm(image_part.view(1, -1) - original_image.view(1, -1), p='fro')

    # Return results
    return {
        "adversarial_image": image_part.clone().detach(),
        "prediction": final_probs.clone().detach(),
        "original_prediction": original_prediction.clone().detach(),
        "label_kld": label_divergence.item(),
        "mse": mse.item(),
        "frob": frob.item()
    }


if __name__ == "__main__":
    # Initialize logger
    try:
        logger = SupabaseLogger()
        print("Supabase logger initialized successfully")
    except Exception as e:
        print(f"Error initializing Supabase logger: {str(e)}")
        print("Continuing without logging...")
        logger = None

    # Load in all models and save to one dictionary
    models = load_all_models()
    for model_name in models.keys():  # double-checking they saved successfully
        print(f"- {model_name}")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move all models to device and freeze params
    for name, model in models.items():
        model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        print(f"Moved {name} to {device} and froze parameters")

    # Load all cases and confirm correct num were printed
    adversarial_cases = load_adversarial_cases()
    print(f"\nLoaded {len(adversarial_cases)} adversarial cases ({len(adversarial_cases)//90} images per digit, {len(adversarial_cases)//10} cases per digit)")
    adversarial_cases = adversarial_cases[:2] # TESTING THIS WITH ONLY 2!!!

    # Adversarial training
    print("\nStarting adversarial training on MLP...")
    for idx, (image, label, digit, target) in enumerate(adversarial_cases):
        print(f"\nCase {idx + 1}/900 (digit {digit})")
        for model_name, model in models.items():
            if model_name == 'mlp':
                results = mlp_advex_train(
                    model=model,
                    image=image,
                    label=label,
                    target=target,
                    device=device
                )
                # Print key metrics from results
                print(f"Final KLD: {results['label_kld']:.4f}")
                print(f"Final MSE: {results['mse']:.4f}")
            else:
                results = auto_hadamard_advex_train(
                    model=model,
                    image=image,
                    label=label,
                    target=target,
                    device=device
                )

            # Log results to supabase database
            if results is not None and logger is not None:
                    try:
                        success = logger.log_result(
                            case_idx=idx,
                            model_name=model_name,
                            image=image,
                            label=label,
                            results=results
                        )
                        if success:
                            print(f"Successfully logged results for {model_name}")
                        else:
                            print(f"Failed to log results for {model_name}")
                    except Exception as e:
                        print(f"Error logging results for {model_name}: {str(e)}")



