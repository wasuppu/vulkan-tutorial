package main

import (
	"log"

	"github.com/go-gl/glfw/v3.3/glfw"
)

const (
	WIDTH  = 800
	HEIGHT = 600
)

type HelloTriangleApplication struct {
	window *glfw.Window
}

func (app *HelloTriangleApplication) Run() (err error) {
	err = app.initWindow()
	if err != nil {
		return err
	}
	app.initVulkan()
	app.mainLoop()
	app.cleanup()
	return nil
}

func (app *HelloTriangleApplication) initWindow() error {
	if err := glfw.Init(); err != nil {
		return err
	}
	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)
	glfw.WindowHint(glfw.Resizable, glfw.False)

	window, err := glfw.CreateWindow(WIDTH, HEIGHT, "Vulkan", nil, nil)
	if err != nil {
		return err
	}
	app.window = window

	return nil
}

func (app *HelloTriangleApplication) initVulkan() error {
	return nil
}

func (app *HelloTriangleApplication) mainLoop() {
	for !app.window.ShouldClose() {
		glfw.PollEvents()
	}
}

func (app *HelloTriangleApplication) cleanup() {
	app.window.Destroy()
	glfw.Terminate()
}

func main() {
	app := &HelloTriangleApplication{}

	err := app.Run()
	if err != nil {
		log.Fatalf("%+v\n", err)
	}
}
