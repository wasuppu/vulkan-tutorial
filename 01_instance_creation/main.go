package main

import (
	"fmt"
	"log"
	"runtime"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/vulkan-go/vulkan"
)

const (
	WIDTH  = 800
	HEIGHT = 600
)

func init() {
	runtime.LockOSThread()
}

type HelloTriangleApplication struct {
	window   *glfw.Window
	instance vk.Instance
}

func (app *HelloTriangleApplication) Run() (err error) {
	err = app.initWindow()
	if err != nil {
		return err
	}
	err = app.initVulkan()
	if err != nil {
		return err
	}

	app.mainLoop()

	defer app.cleanup()
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
	vk.SetGetInstanceProcAddr(glfw.GetVulkanGetInstanceProcAddress())
	if err := vk.Init(); err != nil {
		return err
	}
	return app.createInstance()
}

func (app *HelloTriangleApplication) createInstance() error {
	appInfo := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "Hello Triangle",
		ApplicationVersion: vk.MakeVersion(1, 0, 0),
		PEngineName:        "No Engine",
		EngineVersion:      vk.MakeVersion(1, 0, 0),
		ApiVersion:         vk.ApiVersion10,
	}

	extensions := app.getRequiredExtensions()
	createInfo := vk.InstanceCreateInfo{
		SType:                   vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo:        &appInfo,
		EnabledExtensionCount:   uint32(len(extensions)),
		PpEnabledExtensionNames: extensions,
		EnabledLayerCount:       0,
	}

	var instance vk.Instance
	if vk.CreateInstance(&createInfo, nil, &instance) != vk.Success {
		return fmt.Errorf("failed to create instance")
	}
	app.instance = instance

	return nil
}

func (app *HelloTriangleApplication) getRequiredExtensions() []string {
	glfwExtensions := app.window.GetRequiredInstanceExtensions()
	extensions := []string{}
	for _, extension := range glfwExtensions {
		extensions = append(extensions, vk.ToString([]byte(extension)))
	}
	return extensions
}

func (app *HelloTriangleApplication) mainLoop() {
	for !app.window.ShouldClose() {
		glfw.PollEvents()
	}
}

func (app *HelloTriangleApplication) cleanup() {
	vk.DestroyInstance(app.instance, nil)

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
