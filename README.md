# First Generative Adverserial Netwok

Today I am jumping into the big leagues: I have been learing about GANs from @sirajRaval 
I am still a novice at this and that is why I want to try out the first implementation so that I can get used to the concept of working with this information.

## Credit to: R-Suresh ( I borrowed this implementation from him)

I am using the MNIST dataset on fashion: [here] (https://github.com/zalandoresearch/fashion-mnist)

I hope when you go through the same process as I did in my notebook you will get some ideas on how to work with this kind of neural net.

Good Luck!

These are the major things you need to know about GANs they are composed of two neural nets
1- A generator
2- A Discriminator

## Generator Function:
```
def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        #model.add(Dense(2048))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)
```
This is the neural net that generates information from the dataset.

```
def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        #model.add(Dense(512))
        #model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
```
   The discriminator is the neural network that will try to guess how accurate the information that the generator has made is either fake or real. I have to say this is a simplistic way of implimentation but its so you can get an idea of what this is all about. 

Ideally one can change the arhitecture of the discriminators and generator for particular purposes. That is entirely up to you. How this works is you are able to learn from this architecture.

How to use this:
```
git clone https://github.com/atwine/my_first_GAN.git

cd into the folder and fireup jupyternotebook

```
