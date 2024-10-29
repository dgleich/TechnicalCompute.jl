@time using TechnicalCompute

##
# Load MNIST from MLDatasets and show an image
# Load the MNIST dataset
train_x, train_y = MNIST(split=:train)[:]

# Display the first image in the dataset
img = Gray.(reshape(train_x[:, :, 1], 28, 28))
display(img)

##
clamped_img = clamp.(img + randn(size(img)...)*sqrt(2), 0, 1) 
display(Gray.(clamped_img))

##
mosaicview(Gray.(train_x[:,:,1:16]), nrow=4)

##
image(mosaicview(Gray.(train_x[:,:,1:16]), nrow=4), axis=(;yreversed=true))

##

## 
using GLMakie
# Create a figure where I can select the class
# then generate 16 random images from that class
# and look at 16 perturbed variations of the image
# where I have a slider to control the stdev of the noise
# and a button to generate new images
train_x, train_y = MNIST(split=:train)[:]

fig = Figure(size = (800, 600))

default_class = 0  # Default class to display
class_selection = Observable(default_class)
noise_stdev = Observable(0.1)

# Function to get 16 random images from a specific class
function get_random_images(class_label, n=16)
  indices = findall(x -> x == class_label, train_y)
  selected_indices = rand(indices, n)
  return train_x[:, :, selected_indices]
end

# Function to add noise to an image
function add_noise(img, stdev)
  noise = stdev * randn(size(img))
  return clamp.(img + noise, 0, 1)
end

# Plot initial random images
raw_images = get_random_images(default_class)
imagesview = Observable(mosaic(Gray.(raw_images), nrow=4))

ax = GLMakie.Axis(fig[1:4,1:4], aspect=1, yreversed = true)
hidespines!(ax)
hidedecorations!(ax)
image!(ax, imagesview )

# Button to generate new images
button = Button(fig[6, 1], label = "Generate New Images")

# Dropdown to select class
sg = SliderGrid(fig[5,:], 
  (label = "Class Selection", range = 0:9, startvalue = default_class),
  (label = "Noise StdDev", range = [0,0.1,0.2,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2,2.5,3], startvalue = 0.1)
)

class_selection = sg.sliders[1].value
noise_stdev = sg.sliders[2].value 

# Update images on class selection or button click or slider change
onany(class_selection, button.clicks, noise_stdev) do class_label, _, stdev
  raw_images = get_random_images(class_label)
  #perturbed_images = [add_noise(images[:, :, i], stdev) for i in 1:16]
  perturbed_images = clamp.(raw_images .+ stdev*randn(size(raw_images)...), 0, 1)
  imagesview[] = mosaic(Gray.(Float32.(perturbed_images)), nrow=4)
end

# Display figure
display(fig)