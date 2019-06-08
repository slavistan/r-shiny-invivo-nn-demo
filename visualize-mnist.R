library(keras)
library(shiny)
library(ggrepel)
library(tidyverse)
 
# Load MNIST dataset, normalize and flatten 28x28 images using row-major layout (which is used by tensorflow).
# Also normalize the grayscale values into [0; 1] and one-hot-encode the labels.
mnist   <- dataset_mnist()
x.train <- array_reshape(mnist$train$x, c(nrow(mnist$train$x), 784)) / 255
x.test  <- array_reshape(mnist$test$x, c(nrow(mnist$test$x), 784)) / 255
y.train <- to_categorical(mnist$train$y, 10)
y.test  <- to_categorical(mnist$test$y, 10)

# Return ggplot-heatmap from a row-major 1D vector.
# 
# Kept generic enough to plot arbitrary 2D heatmaps, since ggplot does not have any heatmap geoms and 'image()' rotates
# the image for no healthy reason. Might be reused at some point.
plot_heatmap <- function(data, nrow = 1, ncol = length(data)) {

  X = rep(1:ncol, times = nrow) # X = 123123123
  Y = rep(1:nrow, each  = ncol) # Y = 111222333

  heat.map <- ggplot()                                    +
    geom_raster(aes(x = X, y = Y, fill = data))           +
    coord_cartesian(xlim = c(1, nrow), ylim = c(1, nrow)) +
    theme_void()                                          +
    theme(legend.position="none")

  return(heat.map)
}

# Return untrained Multi-Layer Perceptron Keras model with a single hidden layer.
#
# The number of hidden neurons is parameterized.
initialize_model <- function(input.size = 784, hidden.units = 16, output.size = 10) {

  # Defining is a 2-part process: Firstly, the layers are specified. Secondly, the loss-function and the optimization
  # algorithm are chosed. The second step is referred to as 'compiling' the model in Keras lingo.
  model <- keras_model_sequential(name = "MNIST Classifier")                                                   %>%
    layer_dense(units = hidden.units, activation = 'relu', input_shape = c(input.size), name = "Hidden-Layer") %>%
    layer_dense(units = output.size, activation = 'softmax', name = "Output-Layer")                            %>%
    compile(loss = 'categorical_crossentropy', metrics = c('accuracy'), optimizer = optimizer_rmsprop())

  return(model)
}

### Define the app's user interface
ui <- fluidPage(title = "Live Neural Networks Demo", width="960px",
  fluidRow(
    column(4, offset = 2, align="center",
      h2("Test Set"),
      sliderInput('test.index', label=NULL, min=1, max=10000, value=1000),
      strong(textOutput(outputId = 'label.test')),
      plotOutput(outputId = 'image.test')
    ),
    column(4, align = "center",
      h2("Training Set"),
      sliderInput('train.index', label=NULL, min=1, max=60000, value=8400),
      strong(textOutput(outputId = 'label.train')),
      plotOutput(outputId = 'image.train')
    )
  ),
  fluidRow(
    column(4, align="center",
      sliderInput("epoch.increment", label="Numer of Epochs to Train", value=16, min=1, max=64, step=1),
      splitLayout(cellWidths=c("65%", "35%"),
        actionButton("reset.button", label="Reset + Run", width="100%"),
        actionButton("run.button", label="Run", width="100%")
      ),
      br(),
      splitLayout(
        h5("Hidden neurons: ", style="padding-bottom: -10px;"),
        numericInput("hl.size", label=NULL, min=1, max=128, step=1, value=3)),
      # TODO: Properly concatenate the static prefix and the dynamic counter value
      h4(textOutput("epoch.counter")),
      br(),
      radioButtons("geom.plot.option", label=NULL, choices=list("Point", "Line"), inline=T),
      checkboxInput("show.train.acc.option", label="Show training set history", value=T)
    ),
    column(8, align="center",
      plotOutput("plot.history")
    )
  )
)

### Define the app's reactive outputs, i.e. the 'server'
server <- function(input, output) {

  # Pixmap of test image
  output$image.test <- renderPlot({

   return(plot_heatmap(x.test[input$test.index,], nrow = 28, ncol = 28))
  })
  
  # Pixmap of training image
  output$image.train <- renderPlot({

    return(plot_heatmap(x.train[input$train.index,], nrow = 28, ncol = 28))
  })

  # Label of test image
  output$label.test <- renderText({

    return(paste0("Label: ", mnist$test$y[[input$test.index]]))
  })

  # Label of training image
  output$label.train <- renderText({

    return(paste0("Label: ", mnist$train$y[[input$train.index]]))
  })
}

# Run the app
app <- shinyApp(ui = ui, server = server)
