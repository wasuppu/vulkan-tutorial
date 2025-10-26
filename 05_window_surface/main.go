package main

import (
	"fmt"
	"log"
	"runtime"
	"strings"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/vulkan-go/vulkan"
)

const (
	WIDTH  = 800
	HEIGHT = 600
)

var (
	debug                  = true
	enableValidationLayers = false
	validationLayers       = []string{"VK_LAYER_KHRONOS_validation\x00"}
)

func init() {
	runtime.LockOSThread()
	if debug {
		enableValidationLayers = true
	}
}

type HelloTriangleApplication struct {
	window              *glfw.Window
	instance            vk.Instance
	debugReportCallback vk.DebugReportCallback
	surface             vk.Surface
	physicalDevice      vk.PhysicalDevice
	device              vk.Device
	graphicsQueue       vk.Queue
	presentQueue        vk.Queue
}

func (app *HelloTriangleApplication) Run() (err error) {
	if err = app.initWindow(); err != nil {
		return err
	}

	if err = app.initVulkan(); err != nil {
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

	if err := app.createInstance(); err != nil {
		return err
	}

	if err := app.setupDebugMessenger(); err != nil {
		return err
	}

	if err := app.createSurface(); err != nil {
		return err
	}

	if err := app.pickPhysicalDevice(); err != nil {
		return err
	}

	if err := app.createLogicalDevice(); err != nil {
		return err
	}
	return nil
}

func (app *HelloTriangleApplication) createInstance() error {
	if enableValidationLayers && !checkValidationLayerSupport() {
		return fmt.Errorf("validation layers requested, but not available")
	}

	appInfo := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "Hello Triangle",
		ApplicationVersion: vk.MakeVersion(1, 0, 0),
		PEngineName:        "No Engine",
		EngineVersion:      vk.MakeVersion(1, 0, 0),
		ApiVersion:         vk.ApiVersion10,
	}

	extensions := app.window.GetRequiredInstanceExtensions()
	if enableValidationLayers {
		extensions = append(extensions, "VK_EXT_debug_report\x00")
	}

	createInfo := vk.InstanceCreateInfo{
		SType:                   vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo:        &appInfo,
		EnabledExtensionCount:   uint32(len(extensions)),
		PpEnabledExtensionNames: extensions,
	}

	if enableValidationLayers {
		createInfo.PpEnabledLayerNames = validationLayers
		createInfo.EnabledLayerCount = uint32(len(validationLayers))
		dbgCreateInfo := populateDebugMessengerCreateInfo()
		createInfo.PNext = unsafe.Pointer(dbgCreateInfo.Ref())
	} else {
		createInfo.EnabledLayerCount = 0
		createInfo.PNext = nil
	}

	var instance vk.Instance
	if vk.CreateInstance(&createInfo, nil, &instance) != vk.Success {
		return fmt.Errorf("failed to create instance")
	}
	app.instance = instance

	return nil
}

func (app *HelloTriangleApplication) setupDebugMessenger() error {
	if !enableValidationLayers {
		return nil
	}
	var dbg vk.DebugReportCallback
	dbgCreateInfo := populateDebugMessengerCreateInfo()

	if vk.CreateDebugReportCallback(app.instance, dbgCreateInfo, nil, &dbg) != vk.Success {
		return fmt.Errorf("failed to set up debug messenger")
	}

	app.debugReportCallback = dbg
	return nil
}

func (app *HelloTriangleApplication) createSurface() error {
	surfaceAddr, err := app.window.CreateWindowSurface(app.instance, nil)
	if err != nil {
		return err
	}
	app.surface = vk.SurfaceFromPointer(surfaceAddr)
	return nil
}

func (app *HelloTriangleApplication) pickPhysicalDevice() error {
	var deviceCount uint32
	vk.EnumeratePhysicalDevices(app.instance, &deviceCount, nil)
	if deviceCount == 0 {
		return fmt.Errorf("failed to find GPUs with Vulkan support")
	}
	devices := make([]vk.PhysicalDevice, deviceCount)
	vk.EnumeratePhysicalDevices(app.instance, &deviceCount, devices)

	for _, device := range devices {
		if isDeviceSuitable(device, app.surface) {
			app.physicalDevice = device
			break
		}
	}

	if unsafe.Pointer(app.physicalDevice) == vk.NullHandle {
		return fmt.Errorf("failed to find a suitable gpu")
	}

	return nil
}

func (app *HelloTriangleApplication) createLogicalDevice() error {
	indices := findQueueFamilies(app.physicalDevice, app.surface)

	queueCreateInfos := []vk.DeviceQueueCreateInfo{}
	uniqueQueueFamilies := make(map[uint32]any)
	uniqueQueueFamilies[*indices.graphicsFamily] = nil
	uniqueQueueFamilies[*indices.presentFamily] = nil

	var queuePriority float32 = 1.0
	for queueFamily := range uniqueQueueFamilies {
		queueCreateInfo := vk.DeviceQueueCreateInfo{
			SType:            vk.StructureTypeDeviceQueueCreateInfo,
			QueueFamilyIndex: queueFamily,
			QueueCount:       1,
			PQueuePriorities: []float32{queuePriority},
		}
		queueCreateInfos = append(queueCreateInfos, queueCreateInfo)
	}

	deviceFeatures := []vk.PhysicalDeviceFeatures{{}}
	createInfo := vk.DeviceCreateInfo{
		SType:                 vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount:  uint32(len(queueCreateInfos)),
		PQueueCreateInfos:     queueCreateInfos,
		PEnabledFeatures:      deviceFeatures,
		EnabledExtensionCount: 0,
	}

	if enableValidationLayers {
		createInfo.EnabledLayerCount = uint32(len(validationLayers))
		createInfo.PpEnabledLayerNames = validationLayers
	} else {
		createInfo.EnabledLayerCount = 0
	}

	var device vk.Device
	if vk.CreateDevice(app.physicalDevice, &createInfo, nil, &device) != vk.Success {
		return fmt.Errorf("could not create logical device")
	}
	app.device = device

	var graphicsQueue, presentQueue vk.Queue
	vk.GetDeviceQueue(device, *indices.graphicsFamily, 0, &graphicsQueue)
	vk.GetDeviceQueue(device, *indices.presentFamily, 0, &presentQueue)
	app.graphicsQueue = graphicsQueue
	app.presentQueue = presentQueue

	return nil
}

func (app *HelloTriangleApplication) mainLoop() {
	for !app.window.ShouldClose() {
		glfw.PollEvents()
	}
}

func (app *HelloTriangleApplication) cleanup() {
	vk.DestroyDevice(app.device, nil)

	if enableValidationLayers {
		vk.DestroyDebugReportCallback(app.instance, app.debugReportCallback, nil)
	}

	vk.DestroySurface(app.instance, app.surface, nil)
	vk.DestroyInstance(app.instance, nil)

	app.window.Destroy()
	glfw.Terminate()
}

func isDeviceSuitable(device vk.PhysicalDevice, surface vk.Surface) bool {
	indices := findQueueFamilies(device, surface)
	return indices.isComplete()
}

type QueueFamilyIndices struct {
	graphicsFamily *uint32
	presentFamily  *uint32
}

func (q *QueueFamilyIndices) isComplete() bool {
	return q.graphicsFamily != nil && q.presentFamily != nil
}

func findQueueFamilies(device vk.PhysicalDevice, surface vk.Surface) QueueFamilyIndices {
	var indices QueueFamilyIndices
	var queueFamilyCount uint32
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nil)
	queueFamilies := make([]vk.QueueFamilyProperties, queueFamilyCount)
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies)

	for i, queueFamily := range queueFamilies {
		queueFamily.Deref()
		queueFlags := queueFamily.QueueFlags
		queueFamily.Free()

		if queueFlags&vk.QueueFlags(vk.QueueGraphicsBit) != 0 {
			tmp := uint32(i)
			indices.graphicsFamily = &tmp
		}

		var presentSupport vk.Bool32
		vk.GetPhysicalDeviceSurfaceSupport(device, uint32(i), surface, &presentSupport)
		if presentSupport == vk.True {
			tmp := uint32(i)
			indices.presentFamily = &tmp
		}

		if indices.isComplete() {
			break
		}
	}
	return indices
}

func populateDebugMessengerCreateInfo() *vk.DebugReportCallbackCreateInfo {
	return &vk.DebugReportCallbackCreateInfo{
		SType:       vk.StructureTypeDebugReportCallbackCreateInfo,
		Flags:       vk.DebugReportFlags(vk.DebugReportPerformanceWarningBit | vk.DebugReportWarningBit | vk.DebugReportErrorBit),
		PfnCallback: debugCallback,
		PNext:       nil,
		PUserData:   nil,
	}
}

func debugCallback(flags vk.DebugReportFlags, objectType vk.DebugReportObjectType, object uint64, location uint,
	messageCode int32, layerPrefix string, message string, userData unsafe.Pointer) vk.Bool32 {
	log.Printf("validation layer: %s\n", message)
	return vk.False
}

func checkValidationLayerSupport() bool {
	var layerCount uint32
	vk.EnumerateInstanceLayerProperties(&layerCount, nil)
	availableLayers := make([]vk.LayerProperties, layerCount)
	vk.EnumerateInstanceLayerProperties(&layerCount, availableLayers)

	for _, layerName := range validationLayers {
		layerFound := false

		for _, layerProperties := range availableLayers {
			layerProperties.Deref()
			name := vk.ToString(layerProperties.LayerName[:])
			layerProperties.Free()

			if strings.Trim(layerName, "\x00") == name {
				layerFound = true
				break
			}
		}

		if !layerFound {
			return false
		}
	}
	return true
}

func main() {
	app := &HelloTriangleApplication{}

	err := app.Run()
	if err != nil {
		log.Fatalf("%+v\n", err)
	}
}
