import re
import datetime
import matplotlib.pyplot as plt

# Function to parse log file and extract losses and timestamps
def parse_log_file(log_file):
    epoch = []
    steps = []
    times = []
    perceptual = []
    gan_feat = []
    diversity = []
    generator = []
    discriminator = []

    with open(log_file, 'r') as file:
        for line in file:
            match = re.match(r'Epoch: (\d+), Steps: (\d+), Time: (\d{4}-\d{2}-\d{2})[-\s](\d{2}:\d{2}:\d{2})', line)
            if match:
                epoch.append(int(match.group(1)))
                steps.append(int(match.group(2)))
                time_str = f"{match.group(3)}-{match.group(4)}"
                times.append(datetime.datetime.strptime(time_str, '%Y-%m-%d-%H:%M:%S'))

            match = re.match(r'Perceptual: ([\d.]+), GAN_Feat: ([\d.]+), Diversity: ([\d.]+), Generator: ([\d.]+), Discriminator: ([\d.]+),', line)
            if match:
                perceptual.append(float(match.group(1)))
                gan_feat.append(float(match.group(2)))
                diversity.append(float(match.group(3)))
                generator.append(float(match.group(4)))
                discriminator.append(float(match.group(5)))
    min_len = min(len(epoch), len(steps), len(times), len(perceptual), len(gan_feat), len(diversity), len(generator), len(discriminator))

    epoch = epoch[:min_len]
    steps = steps[:min_len]
    times = times[:min_len]
    perceptual = perceptual[:min_len]
    gan_feat = gan_feat[:min_len]
    diversity = diversity[:min_len]
    generator = generator[:min_len]
    discriminator = discriminator[:min_len]

    return epoch, steps, times, perceptual, gan_feat, diversity, generator, discriminator

# Function to visualize losses with time
def visualize_losses(epoch,steps, times, perceptual, gan_feat, diversity, generator, discriminator):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, perceptual, label='Perceptual')
    plt.plot(steps, gan_feat, label='GAN_Feat')
    plt.plot(steps, diversity, label='Diversity')
    plt.plot(steps, generator, label='Generator')
    plt.plot(steps, discriminator, label='Discriminator')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Losses Over Time')
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# Parse the log file and visualize losses
log_file = 'checkpoints/PDGAN-Training/logs/train.log'
epoch, steps, times, perceptual, gan_feat, diversity, generator, discriminator = parse_log_file(log_file)
visualize_losses(epoch,steps, times, perceptual, gan_feat, diversity, generator, discriminator)