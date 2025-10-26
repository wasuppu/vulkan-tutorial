package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	_ "image/jpeg"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/vulkan-go/vulkan"
)

const (
	WIDTH                = 800
	HEIGHT               = 600
	MAX_FRAMES_IN_FLIGHT = 2
)

var (
	debug                  = true
	enableValidationLayers = false
	validationLayers       = []string{"VK_LAYER_KHRONOS_validation\x00"}
	deviceExtensions       = []string{"VK_KHR_swapchain\x00"}
)

func init() {
	runtime.LockOSThread()
	if debug {
		enableValidationLayers = true
	}
}

type vec2 [2]float32
type vec3 [3]float32
type vec4 [4]float32
type mat4 [4]vec4

func (v vec3) sub(o vec3) vec3 {
	return vec3{v[0] - o[0], v[1] - o[1], v[2] - o[2]}
}

func (v vec3) muln(t float32) vec3 {
	return vec3{v[0] * t, v[1] * t, v[2] * t}
}

func (v vec3) dot(o vec3) float32 {
	return v[0]*o[0] + v[1]*o[1] + v[2]*o[2]
}

func (v vec3) cross(o vec3) vec3 {
	x := v[1]*o[2] - v[2]*o[1]
	y := v[2]*o[0] - v[0]*o[2]
	z := v[0]*o[1] - v[1]*o[0]

	return vec3{x, y, z}
}

func (v vec3) norm() float32 {
	return float32(math.Sqrt(float64(v.dot(v))))
}

func (v vec3) normalize() vec3 {
	return v.muln(float32(1 / v.norm()))
}

func (m mat4) mul(n mat4) mat4 {
	a := mat4{}
	for i := range 4 {
		for j := range 4 {
			for k := range 4 {
				a[i][j] += m[i][k] * n[k][j]
			}
		}
	}
	return a
}

func identity4() mat4 {
	m := mat4{}
	for i := range 4 {
		for j := range 4 {
			if i == j {
				m[i][j] = 1
			} else {
				m[i][j] = 0
			}
		}
	}
	return m
}

func rotate(v vec3, a float32) mat4 {
	v = v.normalize()
	s := float32(math.Sin(float64(a)))
	c := float32(math.Cos(float64(a)))
	m := float32(1 - c)

	return mat4{
		{m*v[0]*v[0] + c, m*v[0]*v[1] + v[2]*s, m*v[2]*v[0] - v[1]*s, 0},
		{m*v[0]*v[1] - v[2]*s, m*v[1]*v[1] + c, m*v[1]*v[2] + v[0]*s, 0},
		{m*v[2]*v[0] + v[1]*s, m*v[1]*v[2] - v[0]*s, m*v[2]*v[2] + c, 0},
		{0, 0, 0, 0}}
}

func (m mat4) rotate(a float32, v vec3) mat4 {
	r := rotate(v, a).mul(m)
	r[3] = m[3]
	return r
}

func lookAt(eye, center, up vec3) mat4 {
	up = up.normalize()
	f := center.sub(eye).normalize()
	s := f.cross(up).normalize()
	u := s.cross(f)

	return mat4{
		{s[0], u[0], -f[0], 0},
		{s[1], u[1], -f[1], 0},
		{s[2], u[2], -f[2], 0},
		{-s.dot(eye), -u.dot(eye), f.dot(eye), 1},
	}
}

func perspective(fovy, aspect, znear, zfar float32) mat4 {
	tanHalfFovy := float32(math.Tan(float64(fovy / 2)))
	m := mat4{}
	m[0][0] = 1 / (aspect * tanHalfFovy)
	m[1][1] = 1 / tanHalfFovy
	m[2][2] = -(zfar + znear) / (zfar - znear)
	m[2][3] = -1
	m[3][2] = -(zfar * znear) / (zfar - znear)
	return m
}

func radians(angle float32) float32 {
	return angle * math.Pi / 180
}

type Vertex struct {
	pos      vec3
	color    vec3
	texCoord vec2
}

func getBindingDescription() vk.VertexInputBindingDescription {
	bindingDescription := vk.VertexInputBindingDescription{
		Binding:   0,
		Stride:    uint32(unsafe.Sizeof(Vertex{})),
		InputRate: vk.VertexInputRateVertex,
	}
	return bindingDescription
}

func getAttributeDescriptions() []vk.VertexInputAttributeDescription {
	attributeDescriptions := make([]vk.VertexInputAttributeDescription, 3)
	attributeDescriptions[0] = vk.VertexInputAttributeDescription{
		Binding:  0,
		Location: 0,
		Format:   vk.FormatR32g32b32Sfloat,
		Offset:   uint32(unsafe.Offsetof(Vertex{}.pos)),
	}

	attributeDescriptions[1] = vk.VertexInputAttributeDescription{
		Binding:  0,
		Location: 1,
		Format:   vk.FormatR32g32b32Sfloat,
		Offset:   uint32(unsafe.Offsetof(Vertex{}.color)),
	}

	attributeDescriptions[2] = vk.VertexInputAttributeDescription{
		Binding:  0,
		Location: 2,
		Format:   vk.FormatR32g32Sfloat,
		Offset:   uint32(unsafe.Offsetof(Vertex{}.texCoord)),
	}

	return attributeDescriptions
}

type UniformBufferObject struct {
	model mat4
	view  mat4
	proj  mat4
}

var vertices = []Vertex{
	{vec3{-0.5, -0.5, 0.0}, vec3{1.0, 0.0, 0.0}, vec2{0.0, 0.0}},
	{vec3{0.5, -0.5, 0.0}, vec3{0.0, 1.0, 0.0}, vec2{1.0, 0.0}},
	{vec3{0.5, 0.5, 0.0}, vec3{0.0, 0.0, 1.0}, vec2{1.0, 1.0}},
	{vec3{-0.5, 0.5, 0.0}, vec3{1.0, 1.0, 1.0}, vec2{0.0, 1.0}},

	{vec3{-0.5, -0.5, -0.5}, vec3{1.0, 0.0, 0.0}, vec2{0.0, 0.0}},
	{vec3{0.5, -0.5, -0.5}, vec3{0.0, 1.0, 0.0}, vec2{1.0, 0.0}},
	{vec3{0.5, 0.5, -0.5}, vec3{0.0, 0.0, 1.0}, vec2{1.0, 1.0}},
	{vec3{-0.5, 0.5, -0.5}, vec3{1.0, 1.0, 1.0}, vec2{0.0, 1.0}},
}

var indices = []uint16{
	0, 1, 2, 2, 3, 0,
	4, 5, 6, 6, 7, 4,
}

type HelloTriangleApplication struct {
	window              *glfw.Window
	instance            vk.Instance
	debugReportCallback vk.DebugReportCallback
	surface             vk.Surface

	physicalDevice vk.PhysicalDevice
	device         vk.Device

	graphicsQueue vk.Queue
	presentQueue  vk.Queue

	swapChain             vk.Swapchain
	swapChainImages       []vk.Image
	swapChainImageFormat  vk.Format
	swapChainExtent       vk.Extent2D
	swapChainImageViews   []vk.ImageView
	swapChainFramebuffers []vk.Framebuffer

	renderPass          vk.RenderPass
	descriptorSetLayout vk.DescriptorSetLayout
	pipelineLayout      vk.PipelineLayout
	graphicsPipeline    vk.Pipeline

	commandPool vk.CommandPool

	depthImage       vk.Image
	depthImageMemory vk.DeviceMemory
	depthImageView   vk.ImageView

	textureImage       vk.Image
	textureImageMemory vk.DeviceMemory
	textureImageView   vk.ImageView
	textureSampler     vk.Sampler

	vertexBuffer           vk.Buffer
	vertexBufferMemory     vk.DeviceMemory
	indexBuffer            vk.Buffer
	indexBufferMemory      vk.DeviceMemory
	uniformBuffers         []vk.Buffer
	uniformBuffersMemories []vk.DeviceMemory
	uniformBuffersMapped   []unsafe.Pointer
	descriptorPool         vk.DescriptorPool
	descriptorSets         []vk.DescriptorSet
	commandBuffers         []vk.CommandBuffer

	imageAvailableSemaphores []vk.Semaphore
	renderFinishedSemaphores []vk.Semaphore
	inFlightFences           []vk.Fence

	currentFrame       uint32
	framebufferResized bool

	startTime time.Time
}

func (app *HelloTriangleApplication) Run() (err error) {
	app.startTime = time.Now()

	if err = app.initWindow(); err != nil {
		return err
	}

	if err = app.initVulkan(); err != nil {
		return err
	}

	if err = app.mainLoop(); err != nil {
		return err
	}

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
	window.SetUserPointer(window.GetUserPointer())
	window.SetFramebufferSizeCallback(func(w *glfw.Window, width int, height int) {
		app.framebufferResized = true
	})

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

	if err := app.createSwapChain(); err != nil {
		return err
	}

	if err := app.createImageViews(); err != nil {
		return err
	}

	if err := app.createRenderPass(); err != nil {
		return err
	}

	if err := app.createDescriptorSetLayout(); err != nil {
		return err
	}

	if err := app.createGraphicsPipeline(); err != nil {
		return err
	}

	if err := app.createCommandPool(); err != nil {
		return err
	}

	if err := app.createDepthResources(); err != nil {
		return err
	}

	if err := app.createFramebuffers(); err != nil {
		return err
	}

	if err := app.createTextureImage(); err != nil {
		return err
	}

	if err := app.createTextureImageView(); err != nil {
		return err
	}

	if err := app.createTextureSampler(); err != nil {
		return err
	}

	if err := app.createVertexBuffer(); err != nil {
		return err
	}

	if err := app.createIndexBuffer(); err != nil {
		return err
	}

	if err := app.createUniformBuffers(); err != nil {
		return err
	}

	if err := app.createDescriptorPool(); err != nil {
		return err
	}

	if err := app.createDescriptorSets(); err != nil {
		return err
	}

	if err := app.createCommandBuffers(); err != nil {
		return err
	}

	if err := app.createSyncObjects(); err != nil {
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

	deviceFeatures := []vk.PhysicalDeviceFeatures{{SamplerAnisotropy: vk.True}}
	createInfo := vk.DeviceCreateInfo{
		SType:                   vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount:    uint32(len(queueCreateInfos)),
		PQueueCreateInfos:       queueCreateInfos,
		PEnabledFeatures:        deviceFeatures,
		EnabledExtensionCount:   uint32(len(deviceExtensions)),
		PpEnabledExtensionNames: deviceExtensions,
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

func (app *HelloTriangleApplication) createSwapChain() error {
	swapChainSupport := querySwapChainSupport(app.physicalDevice, app.surface)
	swapChainSupport.capabilities.Deref()
	defer swapChainSupport.capabilities.Free()

	surfaceFormat := chooseSwapSurfaceFormat(swapChainSupport.formats)
	presentMode := chooseSwapPresentMode(swapChainSupport.presentModes)
	extent := chooseSwapExtent(swapChainSupport.capabilities, app.window)
	imageCount := swapChainSupport.capabilities.MinImageCount + 1

	if swapChainSupport.capabilities.MaxImageCount > 0 && imageCount > swapChainSupport.capabilities.MaxImageCount {
		imageCount = swapChainSupport.capabilities.MaxImageCount
	}

	createInfo := vk.SwapchainCreateInfo{
		SType:            vk.StructureTypeSwapchainCreateInfo,
		Surface:          app.surface,
		MinImageCount:    imageCount,
		ImageFormat:      surfaceFormat.Format,
		ImageColorSpace:  surfaceFormat.ColorSpace,
		ImageExtent:      extent,
		ImageArrayLayers: 1,
		ImageUsage:       vk.ImageUsageFlags(vk.ImageUsageColorAttachmentBit),
		PreTransform:     swapChainSupport.capabilities.CurrentTransform,
		CompositeAlpha:   vk.CompositeAlphaOpaqueBit,
		PresentMode:      presentMode,
		Clipped:          vk.True,
		OldSwapchain:     vk.NullSwapchain,
	}

	indices := findQueueFamilies(app.physicalDevice, app.surface)
	queueFamilyIndices := []uint32{*indices.presentFamily, *indices.graphicsFamily}

	if *indices.graphicsFamily != *indices.presentFamily {
		createInfo.ImageSharingMode = vk.SharingModeConcurrent
		createInfo.QueueFamilyIndexCount = 2
		createInfo.PQueueFamilyIndices = queueFamilyIndices
	} else {
		createInfo.ImageSharingMode = vk.SharingModeExclusive
		createInfo.QueueFamilyIndexCount = 0
		createInfo.PQueueFamilyIndices = nil
	}

	var swapChain vk.Swapchain
	if vk.CreateSwapchain(app.device, &createInfo, nil, &swapChain) != vk.Success {
		return fmt.Errorf("failed to create swap chain")
	}
	app.swapChain = swapChain

	var imagesCount uint32
	vk.GetSwapchainImages(app.device, app.swapChain, &imagesCount, nil)
	app.swapChainImages = make([]vk.Image, imageCount)
	vk.GetSwapchainImages(app.device, app.swapChain, &imagesCount, app.swapChainImages)

	app.swapChainImageFormat = surfaceFormat.Format
	app.swapChainExtent = extent
	return nil
}

func (app *HelloTriangleApplication) createImageViews() error {
	app.swapChainImageViews = make([]vk.ImageView, len(app.swapChainImages))

	for i, image := range app.swapChainImages {
		imageView, err := app.createImageView(image, app.swapChainImageFormat, vk.ImageAspectFlags(vk.ImageAspectColorBit))
		if err != nil {
			return err
		}
		app.swapChainImageViews[i] = imageView
	}

	return nil
}

func (app *HelloTriangleApplication) createRenderPass() error {
	colorAttachment := vk.AttachmentDescription{
		Format:         app.swapChainImageFormat,
		Samples:        vk.SampleCountFlagBits(vk.SampleCount1Bit),
		LoadOp:         vk.AttachmentLoadOpClear,
		StoreOp:        vk.AttachmentStoreOpStore,
		StencilLoadOp:  vk.AttachmentLoadOpDontCare,
		StencilStoreOp: vk.AttachmentStoreOpDontCare,
		InitialLayout:  vk.ImageLayoutUndefined,
		FinalLayout:    vk.ImageLayoutPresentSrc,
	}

	depthFormat, err := app.findDepthFormat()
	if err != nil {
		return err
	}
	depthAttachment := vk.AttachmentDescription{
		Format:         depthFormat,
		Samples:        vk.SampleCount1Bit,
		LoadOp:         vk.AttachmentLoadOpClear,
		StoreOp:        vk.AttachmentStoreOpDontCare,
		StencilLoadOp:  vk.AttachmentLoadOpDontCare,
		StencilStoreOp: vk.AttachmentStoreOpDontCare,
		InitialLayout:  vk.ImageLayoutUndefined,
		FinalLayout:    vk.ImageLayoutDepthStencilAttachmentOptimal,
	}

	colorAttachmentRefs := []vk.AttachmentReference{{
		Attachment: 0,
		Layout:     vk.ImageLayoutColorAttachmentOptimal,
	}}

	depthAttachmentRef := vk.AttachmentReference{
		Attachment: 1,
		Layout:     vk.ImageLayoutDepthStencilAttachmentOptimal,
	}

	subpasses := []vk.SubpassDescription{{
		PipelineBindPoint:       vk.PipelineBindPointGraphics,
		ColorAttachmentCount:    uint32(len(colorAttachmentRefs)),
		PColorAttachments:       colorAttachmentRefs,
		PDepthStencilAttachment: &depthAttachmentRef,
	}}

	dependency := vk.SubpassDependency{
		SrcSubpass:    vk.SubpassExternal,
		DstSubpass:    0,
		SrcStageMask:  vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit | vk.PipelineStageEarlyFragmentTestsBit),
		SrcAccessMask: 0,
		DstStageMask:  vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit | vk.PipelineStageEarlyFragmentTestsBit),
		DstAccessMask: vk.AccessFlags(vk.AccessColorAttachmentWriteBit | vk.AccessDepthStencilAttachmentWriteBit),
	}

	attachments := []vk.AttachmentDescription{colorAttachment, depthAttachment}

	renderPassCreateInfo := vk.RenderPassCreateInfo{
		SType:           vk.StructureTypeRenderPassCreateInfo,
		AttachmentCount: uint32(len(attachments)),
		PAttachments:    attachments,
		SubpassCount:    uint32(len(subpasses)),
		PSubpasses:      subpasses,
		DependencyCount: 1,
		PDependencies:   []vk.SubpassDependency{dependency},
	}

	var renderPass vk.RenderPass

	if vk.CreateRenderPass(app.device, &renderPassCreateInfo, nil, &renderPass) != vk.Success {
		return fmt.Errorf("failed to create render pass")
	}

	app.renderPass = renderPass

	return nil
}

func (app *HelloTriangleApplication) createDescriptorSetLayout() error {
	uboLayoutBinding := vk.DescriptorSetLayoutBinding{
		Binding:            0,
		DescriptorCount:    1,
		DescriptorType:     vk.DescriptorTypeUniformBuffer,
		PImmutableSamplers: nil,
		StageFlags:         vk.ShaderStageFlags(vk.ShaderStageVertexBit),
	}

	samplerLayoutBinding := vk.DescriptorSetLayoutBinding{
		Binding:            1,
		DescriptorCount:    1,
		DescriptorType:     vk.DescriptorTypeCombinedImageSampler,
		PImmutableSamplers: nil,
		StageFlags:         vk.ShaderStageFlags(vk.ShaderStageFragmentBit),
	}

	bindings := []vk.DescriptorSetLayoutBinding{uboLayoutBinding, samplerLayoutBinding}

	layoutInfo := vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}

	var descriptorSetLayout vk.DescriptorSetLayout
	if vk.CreateDescriptorSetLayout(app.device, &layoutInfo, nil, &descriptorSetLayout) != vk.Success {
		return fmt.Errorf("failed to create descriptor set layout")
	}

	app.descriptorSetLayout = descriptorSetLayout
	return nil
}

func (app *HelloTriangleApplication) createGraphicsPipeline() error {
	_, fileName, _, _ := runtime.Caller(0)
	dirpath := filepath.Dir(fileName)

	vertShaderCode, err := os.ReadFile(dirpath + "/shaders/vert.spv")
	if err != nil {
		return err
	}
	fragShaderCode, err := os.ReadFile(dirpath + "/shaders/frag.spv")
	if err != nil {
		return err
	}

	vertShaderModule, err := createShaderModule(vertShaderCode, app.device)
	if err != nil {
		return err
	}
	fragShaderModule, err := createShaderModule(fragShaderCode, app.device)
	if err != nil {
		return err
	}

	vertShaderStageInfo := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageVertexBit,
		Module: vertShaderModule,
		PName:  "main\x00",
	}
	fragShaderStageInfo := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageFragmentBit,
		Module: fragShaderModule,
		PName:  "main\x00",
	}

	shaderStages := []vk.PipelineShaderStageCreateInfo{vertShaderStageInfo, fragShaderStageInfo}

	bindingDescription := getBindingDescription()
	attributeDescriptions := getAttributeDescriptions()
	vertexInputInfo := vk.PipelineVertexInputStateCreateInfo{
		SType:                           vk.StructureTypePipelineVertexInputStateCreateInfo,
		VertexBindingDescriptionCount:   1,
		PVertexBindingDescriptions:      []vk.VertexInputBindingDescription{bindingDescription},
		VertexAttributeDescriptionCount: uint32(len(attributeDescriptions)),
		PVertexAttributeDescriptions:    attributeDescriptions,
	}

	inputAssembly := vk.PipelineInputAssemblyStateCreateInfo{
		SType:                  vk.StructureTypePipelineInputAssemblyStateCreateInfo,
		Topology:               vk.PrimitiveTopologyTriangleList,
		PrimitiveRestartEnable: vk.False,
	}

	viewportState := vk.PipelineViewportStateCreateInfo{
		SType:         vk.StructureTypePipelineViewportStateCreateInfo,
		ViewportCount: 1,
		ScissorCount:  1,
	}

	rasterizer := vk.PipelineRasterizationStateCreateInfo{
		SType:                   vk.StructureTypePipelineRasterizationStateCreateInfo,
		DepthClampEnable:        vk.False,
		RasterizerDiscardEnable: vk.False,
		PolygonMode:             vk.PolygonModeFill,
		LineWidth:               1.0,
		CullMode:                vk.CullModeFlags(vk.CullModeBackBit),
		FrontFace:               vk.FrontFaceCounterClockwise,
		DepthBiasEnable:         vk.False,
	}

	multisampling := vk.PipelineMultisampleStateCreateInfo{
		SType:                vk.StructureTypePipelineMultisampleStateCreateInfo,
		SampleShadingEnable:  vk.False,
		RasterizationSamples: vk.SampleCount1Bit,
	}

	depthStencil := vk.PipelineDepthStencilStateCreateInfo{
		SType:                 vk.StructureTypePipelineDepthStencilStateCreateInfo,
		DepthTestEnable:       vk.True,
		DepthWriteEnable:      vk.True,
		DepthCompareOp:        vk.CompareOpLess,
		DepthBoundsTestEnable: vk.False,
		MinDepthBounds:        0,
		MaxDepthBounds:        1,
		StencilTestEnable:     vk.False,
	}

	colorBlendAttachments := []vk.PipelineColorBlendAttachmentState{{
		ColorWriteMask: vk.ColorComponentFlags(vk.ColorComponentRBit | vk.ColorComponentGBit | vk.ColorComponentBBit | vk.ColorComponentABit),
		BlendEnable:    vk.False,
	}}

	colorBlending := vk.PipelineColorBlendStateCreateInfo{
		SType:           vk.StructureTypePipelineColorBlendStateCreateInfo,
		LogicOpEnable:   vk.False,
		LogicOp:         vk.LogicOpCopy,
		AttachmentCount: 1,
		PAttachments:    colorBlendAttachments,
		BlendConstants:  [4]float32{0.0, 0.0, 0.0, 0.0},
	}

	dynamicStates := []vk.DynamicState{
		vk.DynamicStateViewport,
		vk.DynamicStateScissor,
	}

	dynamicState := vk.PipelineDynamicStateCreateInfo{
		SType:             vk.StructureTypePipelineDynamicStateCreateInfo,
		DynamicStateCount: uint32(len(dynamicStates)),
		PDynamicStates:    dynamicStates,
	}

	if app.descriptorSetLayout == nil {
		panic("app.descriptorSetLayout")
	}

	pipelineLayoutInfo := vk.PipelineLayoutCreateInfo{
		SType:          vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount: 1,
		PSetLayouts:    []vk.DescriptorSetLayout{app.descriptorSetLayout},
	}

	var pipelineLayout vk.PipelineLayout
	if vk.CreatePipelineLayout(app.device, &pipelineLayoutInfo, nil, &pipelineLayout) != vk.Success {
		return fmt.Errorf("failed to create pipeline layout")
	}
	app.pipelineLayout = pipelineLayout

	pipelineInfos := []vk.GraphicsPipelineCreateInfo{{
		SType:               vk.StructureTypeGraphicsPipelineCreateInfo,
		StageCount:          2,
		PStages:             shaderStages,
		PVertexInputState:   &vertexInputInfo,
		PInputAssemblyState: &inputAssembly,
		PViewportState:      &viewportState,
		PRasterizationState: &rasterizer,
		PMultisampleState:   &multisampling,
		PDepthStencilState:  &depthStencil,
		PColorBlendState:    &colorBlending,
		PDynamicState:       &dynamicState,
		Layout:              app.pipelineLayout,
		RenderPass:          app.renderPass,
		Subpass:             0,
		BasePipelineHandle:  vk.Pipeline(vk.NullHandle),
	}}

	var graphicsPipelines = make([]vk.Pipeline, 1)

	if vk.CreateGraphicsPipelines(app.device, vk.PipelineCache(vk.NullHandle), uint32(len(pipelineInfos)), pipelineInfos, nil, graphicsPipelines) != vk.Success {
		return fmt.Errorf("failed to create graphics pipeline")
	}

	app.graphicsPipeline = graphicsPipelines[0]

	vk.DestroyShaderModule(app.device, fragShaderModule, nil)
	vk.DestroyShaderModule(app.device, vertShaderModule, nil)
	return nil
}

func (app *HelloTriangleApplication) createFramebuffers() error {
	app.swapChainFramebuffers = make([]vk.Framebuffer, len(app.swapChainImageViews))

	for i := range app.swapChainImageViews {
		attachments := []vk.ImageView{
			app.swapChainImageViews[i],
			app.depthImageView,
		}

		fbCreateInfo := vk.FramebufferCreateInfo{
			SType:           vk.StructureTypeFramebufferCreateInfo,
			RenderPass:      app.renderPass,
			AttachmentCount: uint32(len(attachments)),
			PAttachments:    attachments,
			Width:           app.swapChainExtent.Width,
			Height:          app.swapChainExtent.Height,
			Layers:          1,
		}

		var framebuffer vk.Framebuffer
		if vk.CreateFramebuffer(app.device, &fbCreateInfo, nil, &framebuffer) != vk.Success {
			return fmt.Errorf("failed to create framebuffer")
		}

		app.swapChainFramebuffers[i] = framebuffer
	}

	return nil
}

func (app *HelloTriangleApplication) createCommandPool() error {
	queueFamilyIndices := findQueueFamilies(app.physicalDevice, app.surface)

	poolInfo := vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		Flags:            vk.CommandPoolCreateFlags(vk.CommandPoolCreateResetCommandBufferBit),
		QueueFamilyIndex: *queueFamilyIndices.graphicsFamily,
	}
	var commandPool vk.CommandPool
	if vk.CreateCommandPool(app.device, &poolInfo, nil, &commandPool) != vk.Success {
		return fmt.Errorf("failed to create command pool")
	}
	app.commandPool = commandPool
	return nil
}

func (app *HelloTriangleApplication) createDepthResources() error {
	depthFormat, err := app.findDepthFormat()
	if err != nil {
		return err
	}
	var depthImage vk.Image
	var depthImageMemory vk.DeviceMemory

	err = app.createImage(app.swapChainExtent.Width, app.swapChainExtent.Height, depthFormat, vk.ImageTilingOptimal,
		vk.ImageUsageFlags(vk.ImageUsageDepthStencilAttachmentBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyDeviceLocalBit),
		&depthImage, &depthImageMemory)
	if err != nil {
		return err
	}
	app.depthImage = depthImage
	app.depthImageMemory = depthImageMemory

	depthImageView, err := app.createImageView(depthImage, depthFormat, vk.ImageAspectFlags(vk.ImageAspectDepthBit))
	if err != nil {
		return err
	}
	app.depthImageView = depthImageView

	return nil
}

func (app *HelloTriangleApplication) findDepthFormat() (vk.Format, error) {
	return app.findSupportedFormat(
		[]vk.Format{
			vk.FormatD32Sfloat,
			vk.FormatD32SfloatS8Uint,
			vk.FormatD24UnormS8Uint,
		},
		vk.ImageTilingOptimal,
		vk.FormatFeatureFlags(vk.FormatFeatureDepthStencilAttachmentBit),
	)
}

func (app *HelloTriangleApplication) findSupportedFormat(candidates []vk.Format, tiling vk.ImageTiling,
	features vk.FormatFeatureFlags) (vk.Format, error) {

	for _, format := range candidates {
		var props vk.FormatProperties
		vk.GetPhysicalDeviceFormatProperties(app.physicalDevice, format, &props)
		props.Deref()
		defer props.Free()

		if tiling == vk.ImageTilingLinear && (props.LinearTilingFeatures&features) == features {
			return format, nil
		} else if tiling == vk.ImageTilingOptimal && (props.OptimalTilingFeatures&features) == features {
			return format, nil
		}
	}

	return 0, fmt.Errorf("failed to find supported format")
}

func (app *HelloTriangleApplication) createTextureImage() error {
	_, fileName, _, _ := runtime.Caller(0)
	dirpath := filepath.Dir(fileName)

	imageBytes, err := os.ReadFile(dirpath + "/textures/texture.jpg")
	if err != nil {
		return err
	}

	decodedImage, _, err := image.Decode(bytes.NewBuffer(imageBytes))
	if err != nil {
		return err
	}

	bounds := decodedImage.Bounds()
	texWidth := uint32(bounds.Size().X)
	texHeight := uint32(bounds.Size().Y)
	imageSize := vk.DeviceSize(texWidth * texHeight * 4)

	var pixels []byte
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, a := decodedImage.At(x, y).RGBA()
			// https://go.dev/blog/image Colors and Color Models
			// Second, the channels have a 16-bit effective range
			// Third, the type returned is uint32, even though the maximum value is 65535
			// Nevertheless, Vulkan VK_FORM-R8G8B8A8_SRGB format requires each channel to be 8-bit
			pixels = append(pixels, byte(r>>8), byte(g>>8), byte(b>>8), byte(a>>8))
		}
	}

	var stagingBuffer vk.Buffer
	var stagingBufferMemory vk.DeviceMemory
	err = app.createBuffer(imageSize, vk.BufferUsageFlags(vk.BufferUsageTransferSrcBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyHostVisibleBit|vk.MemoryPropertyHostCoherentBit),
		&stagingBuffer, &stagingBufferMemory)
	if err != nil {
		return err
	}

	var data unsafe.Pointer
	vk.MapMemory(app.device, stagingBufferMemory, 0, imageSize, 0, &data)
	memcopy(data, pixels)
	vk.UnmapMemory(app.device, stagingBufferMemory)

	var textureImage vk.Image
	var textureImageMemory vk.DeviceMemory
	err = app.createImage(texWidth, texHeight,
		vk.FormatR8g8b8a8Srgb, vk.ImageTilingOptimal,
		vk.ImageUsageFlags(vk.ImageUsageTransferDstBit|vk.ImageUsageSampledBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyDeviceLocalBit),
		&textureImage, &textureImageMemory)
	if err != nil {
		return err
	}
	app.textureImage = textureImage
	app.textureImageMemory = textureImageMemory

	app.transitionImageLayout(app.textureImage, vk.FormatR8g8b8a8Srgb, vk.ImageLayoutUndefined, vk.ImageLayoutTransferDstOptimal)
	app.copyBufferToImage(stagingBuffer, app.textureImage, texWidth, texHeight)
	app.transitionImageLayout(app.textureImage, vk.FormatR8g8b8a8Srgb, vk.ImageLayoutTransferDstOptimal, vk.ImageLayoutShaderReadOnlyOptimal)

	vk.DestroyBuffer(app.device, stagingBuffer, nil)
	vk.FreeMemory(app.device, stagingBufferMemory, nil)
	return nil
}

func (app *HelloTriangleApplication) createImage(width, height uint32,
	format vk.Format, tiling vk.ImageTiling, usage vk.ImageUsageFlags,
	properties vk.MemoryPropertyFlags, image *vk.Image, imageMemory *vk.DeviceMemory) error {

	imageInfo := vk.ImageCreateInfo{
		SType:     vk.StructureTypeImageCreateInfo,
		ImageType: vk.ImageType2d,
		Extent: vk.Extent3D{
			Width:  width,
			Height: height,
			Depth:  1,
		},
		MipLevels:     1,
		ArrayLayers:   1,
		Format:        format,
		Tiling:        tiling,
		InitialLayout: vk.ImageLayoutUndefined,
		Usage:         usage,
		Samples:       vk.SampleCount1Bit,
		SharingMode:   vk.SharingModeExclusive,
	}

	if vk.CreateImage(app.device, &imageInfo, nil, image) != vk.Success {
		return fmt.Errorf("failed to create an image")
	}

	var memRequirements vk.MemoryRequirements
	vk.GetImageMemoryRequirements(app.device, *image, &memRequirements)
	memRequirements.Deref()
	defer memRequirements.Free()

	memTypeIndex, err := findMemoryType(memRequirements.MemoryTypeBits, properties, app.physicalDevice)
	if err != nil {
		return err
	}

	allocInfo := vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  memRequirements.Size,
		MemoryTypeIndex: memTypeIndex,
	}

	if vk.AllocateMemory(app.device, &allocInfo, nil, imageMemory) != vk.Success {
		return fmt.Errorf("failed to allocate image buffer memory")
	}

	vk.BindImageMemory(app.device, *image, *imageMemory, 0)

	return nil
}

func (app *HelloTriangleApplication) transitionImageLayout(image vk.Image, format vk.Format, oldLayout vk.ImageLayout, newLayout vk.ImageLayout) error {
	commandBuffer := app.beginSingleTimeCommands()

	barrier := vk.ImageMemoryBarrier{
		SType:               vk.StructureTypeImageMemoryBarrier,
		OldLayout:           oldLayout,
		NewLayout:           newLayout,
		SrcQueueFamilyIndex: vk.QueueFamilyIgnored,
		DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		Image:               image,
		SubresourceRange: vk.ImageSubresourceRange{
			AspectMask:     vk.ImageAspectFlags(vk.ImageAspectColorBit),
			BaseMipLevel:   0,
			LevelCount:     1,
			BaseArrayLayer: 0,
			LayerCount:     1,
		},
	}

	var sourceStage vk.PipelineStageFlags
	var destinationStage vk.PipelineStageFlags

	if oldLayout == vk.ImageLayoutUndefined && newLayout == vk.ImageLayoutTransferDstOptimal {
		barrier.SrcAccessMask = 0
		barrier.DstAccessMask = vk.AccessFlags(vk.AccessTransferWriteBit)

		sourceStage = vk.PipelineStageFlags(vk.PipelineStageTopOfPipeBit)
		destinationStage = vk.PipelineStageFlags(vk.PipelineStageTransferBit)
	} else if oldLayout == vk.ImageLayoutTransferDstOptimal && newLayout == vk.ImageLayoutShaderReadOnlyOptimal {
		barrier.SrcAccessMask = vk.AccessFlags(vk.AccessTransferWriteBit)
		barrier.DstAccessMask = vk.AccessFlags(vk.AccessShaderReadBit)

		sourceStage = vk.PipelineStageFlags(vk.PipelineStageTransferBit)
		destinationStage = vk.PipelineStageFlags(vk.PipelineStageFragmentShaderBit)
	} else {
		return fmt.Errorf("unsupported layout transition")
	}

	vk.CmdPipelineBarrier(
		commandBuffer,
		sourceStage, destinationStage,
		0,
		0, nil,
		0, nil,
		1, []vk.ImageMemoryBarrier{barrier},
	)

	app.endSingleTimeCommands(commandBuffer)
	return nil
}

func (app *HelloTriangleApplication) copyBufferToImage(buffer vk.Buffer, image vk.Image, width, height uint32) {
	commandBuffer := app.beginSingleTimeCommands()

	region := vk.BufferImageCopy{
		BufferOffset:      0,
		BufferRowLength:   0,
		BufferImageHeight: 0,

		ImageSubresource: vk.ImageSubresourceLayers{
			AspectMask:     vk.ImageAspectFlags(vk.ImageAspectColorBit),
			MipLevel:       0,
			BaseArrayLayer: 0,
			LayerCount:     1,
		},

		ImageOffset: vk.Offset3D{
			X: 0, Y: 0, Z: 0,
		},

		ImageExtent: vk.Extent3D{
			Width:  width,
			Height: height,
			Depth:  1,
		},
	}

	vk.CmdCopyBufferToImage(commandBuffer, buffer, image,
		vk.ImageLayoutTransferDstOptimal, 1, []vk.BufferImageCopy{region})

	app.endSingleTimeCommands(commandBuffer)
}

func (app *HelloTriangleApplication) createTextureImageView() error {
	var err error
	app.textureImageView, err = app.createImageView(app.textureImage, vk.FormatR8g8b8a8Srgb, vk.ImageAspectFlags(vk.ImageAspectColorBit))
	return err
}

func (app *HelloTriangleApplication) createImageView(image vk.Image, format vk.Format, aspectFlags vk.ImageAspectFlags) (vk.ImageView, error) {
	viewInfo := vk.ImageViewCreateInfo{
		SType:    vk.StructureTypeImageViewCreateInfo,
		Image:    image,
		ViewType: vk.ImageViewType2d,
		Format:   format,
		SubresourceRange: vk.ImageSubresourceRange{
			AspectMask:     aspectFlags,
			BaseMipLevel:   0,
			LevelCount:     1,
			BaseArrayLayer: 0,
			LayerCount:     1,
		},
	}
	var imageView vk.ImageView

	if vk.CreateImageView(app.device, &viewInfo, nil, &imageView) != vk.Success {
		return nil, fmt.Errorf("failed to create texture image view")
	}

	return imageView, nil
}

func (app *HelloTriangleApplication) createTextureSampler() error {
	var properties vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(app.physicalDevice, &properties)
	properties.Deref()
	properties.Limits.Deref()
	defer properties.Limits.Free()
	defer properties.Free()

	samplerInfo := vk.SamplerCreateInfo{
		SType:                   vk.StructureTypeSamplerCreateInfo,
		MagFilter:               vk.FilterLinear,
		MinFilter:               vk.FilterLinear,
		AddressModeU:            vk.SamplerAddressModeRepeat,
		AddressModeV:            vk.SamplerAddressModeRepeat,
		AddressModeW:            vk.SamplerAddressModeRepeat,
		AnisotropyEnable:        vk.True,
		MaxAnisotropy:           properties.Limits.MaxSamplerAnisotropy,
		BorderColor:             vk.BorderColorIntOpaqueBlack,
		UnnormalizedCoordinates: vk.False,
		CompareEnable:           vk.False,
		CompareOp:               vk.CompareOpAlways,
		MipmapMode:              vk.SamplerMipmapModeLinear,
		MipLodBias:              0,
		MinLod:                  0,
		MaxLod:                  0,
	}

	var textureSampler vk.Sampler
	if vk.CreateSampler(app.device, &samplerInfo, nil, &textureSampler) != vk.Success {
		return fmt.Errorf("failed to create texture sampler")
	}
	app.textureSampler = textureSampler

	return nil
}

func (app *HelloTriangleApplication) createVertexBuffer() error {
	bufferSize := vk.DeviceSize(binary.Size(vertices))

	var stagingBuffer vk.Buffer
	var stagingBufferMemory vk.DeviceMemory
	err := app.createBuffer(bufferSize,
		vk.BufferUsageFlags(vk.BufferUsageTransferSrcBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyHostVisibleBit|vk.MemoryPropertyHostCoherentBit),
		&stagingBuffer, &stagingBufferMemory)
	if err != nil {
		return err
	}

	var data unsafe.Pointer
	vk.MapMemory(app.device, stagingBufferMemory, 0, bufferSize, 0, &data)
	memcopy(data, vertices)
	vk.UnmapMemory(app.device, stagingBufferMemory)

	var vertexBuffer vk.Buffer
	var vertexBufferMemory vk.DeviceMemory
	err = app.createBuffer(bufferSize,
		vk.BufferUsageFlags(vk.BufferUsageTransferDstBit|vk.BufferUsageVertexBufferBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyDeviceLocalBit),
		&vertexBuffer, &vertexBufferMemory)
	if err != nil {
		return err
	}

	app.vertexBuffer = vertexBuffer
	app.vertexBufferMemory = vertexBufferMemory

	app.copyBuffer(stagingBuffer, app.vertexBuffer, bufferSize)

	vk.DestroyBuffer(app.device, stagingBuffer, nil)
	vk.FreeMemory(app.device, stagingBufferMemory, nil)

	return nil
}

func (app *HelloTriangleApplication) createIndexBuffer() error {
	bufferSize := vk.DeviceSize(binary.Size(indices))

	var stagingBuffer vk.Buffer
	var stagingBufferMemory vk.DeviceMemory
	err := app.createBuffer(bufferSize,
		vk.BufferUsageFlags(vk.BufferUsageTransferSrcBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyHostVisibleBit|vk.MemoryPropertyHostCoherentBit),
		&stagingBuffer, &stagingBufferMemory)
	if err != nil {
		return err
	}

	var data unsafe.Pointer
	vk.MapMemory(app.device, stagingBufferMemory, 0, bufferSize, 0, &data)
	memcopy(data, indices)
	vk.UnmapMemory(app.device, stagingBufferMemory)

	var indexBuffer vk.Buffer
	var indexBufferMemory vk.DeviceMemory
	err = app.createBuffer(bufferSize,
		vk.BufferUsageFlags(vk.BufferUsageTransferDstBit|vk.BufferUsageIndexBufferBit),
		vk.MemoryPropertyFlags(vk.MemoryPropertyDeviceLocalBit),
		&indexBuffer, &indexBufferMemory)
	if err != nil {
		return err
	}
	app.indexBuffer = indexBuffer
	app.indexBufferMemory = indexBufferMemory

	app.copyBuffer(stagingBuffer, app.indexBuffer, bufferSize)

	vk.DestroyBuffer(app.device, stagingBuffer, nil)
	vk.FreeMemory(app.device, stagingBufferMemory, nil)

	return nil
}

func (app *HelloTriangleApplication) createUniformBuffers() error {
	bufferSize := vk.DeviceSize(unsafe.Sizeof(UniformBufferObject{}))

	app.uniformBuffers = make([]vk.Buffer, MAX_FRAMES_IN_FLIGHT)
	app.uniformBuffersMemories = make([]vk.DeviceMemory, MAX_FRAMES_IN_FLIGHT)
	app.uniformBuffersMapped = make([]unsafe.Pointer, MAX_FRAMES_IN_FLIGHT)

	for i := range MAX_FRAMES_IN_FLIGHT {
		var buffer vk.Buffer
		var bufferMemory vk.DeviceMemory
		err := app.createBuffer(bufferSize,
			vk.BufferUsageFlags(vk.BufferUsageUniformBufferBit),
			vk.MemoryPropertyFlags(vk.MemoryPropertyHostVisibleBit|vk.MemoryPropertyHostCoherentBit),
			&buffer, &bufferMemory)
		if err != nil {
			return err
		}

		app.uniformBuffers[i] = buffer
		app.uniformBuffersMemories[i] = bufferMemory

		var data unsafe.Pointer
		vk.MapMemory(app.device, app.uniformBuffersMemories[i], 0, bufferSize, 0, &data)
		app.uniformBuffersMapped[i] = data
	}
	return nil
}

func (app *HelloTriangleApplication) createDescriptorPool() error {
	poolSizes := []vk.DescriptorPoolSize{
		{
			Type:            vk.DescriptorTypeUniformBuffer,
			DescriptorCount: MAX_FRAMES_IN_FLIGHT,
		},
		{
			Type:            vk.DescriptorTypeCombinedImageSampler,
			DescriptorCount: MAX_FRAMES_IN_FLIGHT,
		},
	}

	poolInfo := vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		PoolSizeCount: uint32(len(poolSizes)),
		PPoolSizes:    poolSizes,
		MaxSets:       MAX_FRAMES_IN_FLIGHT,
	}

	var descriptorPool vk.DescriptorPool
	if vk.CreateDescriptorPool(app.device, &poolInfo, nil, &descriptorPool) != vk.Success {
		return fmt.Errorf("failed to create descriptor pool")
	}
	app.descriptorPool = descriptorPool

	return nil
}

func (app *HelloTriangleApplication) createDescriptorSets() error {
	layouts := make([]vk.DescriptorSetLayout, MAX_FRAMES_IN_FLIGHT)
	for i := range MAX_FRAMES_IN_FLIGHT {
		layouts[i] = app.descriptorSetLayout
	}

	allocInfo := vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     app.descriptorPool,
		DescriptorSetCount: MAX_FRAMES_IN_FLIGHT,
		PSetLayouts:        layouts,
	}

	app.descriptorSets = make([]vk.DescriptorSet, MAX_FRAMES_IN_FLIGHT)
	if vk.AllocateDescriptorSets(app.device, &allocInfo, &app.descriptorSets[0]) != vk.Success {
		return fmt.Errorf("failed to allocate descriptor set")
	}

	for i := range MAX_FRAMES_IN_FLIGHT {
		bufferInfo := vk.DescriptorBufferInfo{
			Buffer: app.uniformBuffers[i],
			Offset: 0,
			Range:  vk.DeviceSize(unsafe.Sizeof(UniformBufferObject{})),
		}

		imageInfo := vk.DescriptorImageInfo{
			ImageLayout: vk.ImageLayoutShaderReadOnlyOptimal,
			ImageView:   app.textureImageView,
			Sampler:     app.textureSampler,
		}

		descriptorWrites := []vk.WriteDescriptorSet{
			{
				SType:           vk.StructureTypeWriteDescriptorSet,
				DstSet:          app.descriptorSets[i],
				DstBinding:      0,
				DstArrayElement: 0,
				DescriptorType:  vk.DescriptorTypeUniformBuffer,
				DescriptorCount: 1,
				PBufferInfo:     []vk.DescriptorBufferInfo{bufferInfo},
			},
			{
				SType:           vk.StructureTypeWriteDescriptorSet,
				DstSet:          app.descriptorSets[i],
				DstBinding:      1,
				DstArrayElement: 0,
				DescriptorType:  vk.DescriptorTypeCombinedImageSampler,
				DescriptorCount: 1,
				PImageInfo:      []vk.DescriptorImageInfo{imageInfo},
			}}

		vk.UpdateDescriptorSets(app.device, uint32(len(descriptorWrites)), descriptorWrites, 0, nil)
	}

	return nil
}

func (app *HelloTriangleApplication) createCommandBuffers() error {
	commandBuffers := make([]vk.CommandBuffer, MAX_FRAMES_IN_FLIGHT)

	allocInfo := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        app.commandPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: uint32(len(commandBuffers)),
	}

	if vk.AllocateCommandBuffers(app.device, &allocInfo, commandBuffers) != vk.Success {
		return fmt.Errorf("failed to allocate command buffers")
	}
	app.commandBuffers = commandBuffers
	return nil
}

func (app *HelloTriangleApplication) createSyncObjects() error {
	semaphoreInfo := vk.SemaphoreCreateInfo{
		SType: vk.StructureTypeSemaphoreCreateInfo,
	}

	fenceInfo := vk.FenceCreateInfo{
		SType: vk.StructureTypeFenceCreateInfo,
		Flags: vk.FenceCreateFlags(vk.FenceCreateSignaledBit),
	}

	app.imageAvailableSemaphores = make([]vk.Semaphore, MAX_FRAMES_IN_FLIGHT)
	app.renderFinishedSemaphores = make([]vk.Semaphore, MAX_FRAMES_IN_FLIGHT)
	app.inFlightFences = make([]vk.Fence, MAX_FRAMES_IN_FLIGHT)
	for i := range MAX_FRAMES_IN_FLIGHT {
		var imageAvailableSemaphore, renderFinishedSemaphore vk.Semaphore
		var inFlightFence vk.Fence
		if vk.CreateSemaphore(app.device, &semaphoreInfo, nil, &imageAvailableSemaphore) != vk.Success ||
			vk.CreateSemaphore(app.device, &semaphoreInfo, nil, &renderFinishedSemaphore) != vk.Success ||
			vk.CreateFence(app.device, &fenceInfo, nil, &inFlightFence) != vk.Success {
			return fmt.Errorf("failed to create synchronization objects for a frame")
		}

		app.imageAvailableSemaphores[i] = imageAvailableSemaphore
		app.renderFinishedSemaphores[i] = renderFinishedSemaphore
		app.inFlightFences[i] = inFlightFence
	}

	return nil
}

func (app *HelloTriangleApplication) mainLoop() error {
	for !app.window.ShouldClose() {
		glfw.PollEvents()
		if err := app.drawFrame(); err != nil {
			return err
		}
	}

	vk.DeviceWaitIdle(app.device)
	return nil
}

func (app *HelloTriangleApplication) drawFrame() error {
	vk.WaitForFences(app.device, 1, []vk.Fence{app.inFlightFences[app.currentFrame]}, vk.True, vk.MaxUint64)

	var imageIndex uint32
	result := vk.AcquireNextImage(app.device, app.swapChain, vk.MaxUint64, app.imageAvailableSemaphores[app.currentFrame], vk.Fence(vk.NullHandle), &imageIndex)

	if result == vk.ErrorOutOfDate {
		return app.recreateSwapChain()
	} else if result != vk.Success && result != vk.Suboptimal {
		return fmt.Errorf("failed to acquire swap chain image")
	}

	app.updateUniformBuffer(app.currentFrame)

	vk.ResetFences(app.device, 1, []vk.Fence{app.inFlightFences[app.currentFrame]})

	vk.ResetCommandBuffer(app.commandBuffers[app.currentFrame], 0)
	app.recordCommandBuffer(app.commandBuffers[app.currentFrame], imageIndex)

	waitSemaphores := []vk.Semaphore{app.imageAvailableSemaphores[app.currentFrame]}
	waitStages := []vk.PipelineStageFlags{vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit)}

	commandBuffers := []vk.CommandBuffer{app.commandBuffers[app.currentFrame]}
	signalSemaphores := []vk.Semaphore{app.renderFinishedSemaphores[app.currentFrame]}

	submitInfos := []vk.SubmitInfo{{
		SType:                vk.StructureTypeSubmitInfo,
		WaitSemaphoreCount:   uint32(len(waitSemaphores)),
		PWaitSemaphores:      waitSemaphores,
		PWaitDstStageMask:    waitStages,
		CommandBufferCount:   uint32(len(commandBuffers)),
		PCommandBuffers:      commandBuffers,
		SignalSemaphoreCount: uint32(len(signalSemaphores)),
		PSignalSemaphores:    signalSemaphores,
	}}

	if vk.QueueSubmit(app.graphicsQueue, 1, submitInfos, app.inFlightFences[app.currentFrame]) != vk.Success {
		return fmt.Errorf("failed to submit draw command buffer")
	}

	swapChains := []vk.Swapchain{app.swapChain}
	presentInfo := vk.PresentInfo{
		SType:              vk.StructureTypePresentInfo,
		WaitSemaphoreCount: 1,
		PWaitSemaphores:    signalSemaphores,
		SwapchainCount:     uint32(len(swapChains)),
		PSwapchains:        swapChains,
		PImageIndices:      []uint32{imageIndex},
	}

	result = vk.QueuePresent(app.presentQueue, &presentInfo)
	if result == vk.ErrorOutOfDate || result == vk.Suboptimal || app.framebufferResized {
		app.framebufferResized = false
		return app.recreateSwapChain()
	} else if result != vk.Success {
		return fmt.Errorf("failed to present swap chain image")
	}

	app.currentFrame = (app.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT
	return nil
}

func (app *HelloTriangleApplication) updateUniformBuffer(currentImage uint32) {
	frameTime := time.Since(app.startTime)
	ubo := UniformBufferObject{}

	ubo.model = identity4().rotate(float32(frameTime.Seconds())*radians(90), vec3{0, 0, 1})
	ubo.view = lookAt(vec3{2, 2, 2}, vec3{0, 0, 0}, vec3{0, 0, 1})
	ubo.proj = perspective(radians(45), float32(app.swapChainExtent.Width)/float32(app.swapChainExtent.Height), 0.1, 10)
	ubo.proj[1][1] *= -1

	memcopy(app.uniformBuffersMapped[currentImage], ubo)
}

func (app *HelloTriangleApplication) recreateSwapChain() error {
	width, height := app.window.GetFramebufferSize()
	for width == 0 || height == 0 {
		width, height = app.window.GetFramebufferSize()
		glfw.WaitEvents()
	}
	vk.DeviceWaitIdle(app.device)
	app.cleanupSwapChain()

	if err := app.createSwapChain(); err != nil {
		return err
	}

	if err := app.createImageViews(); err != nil {
		return err
	}

	if err := app.createDepthResources(); err != nil {
		return err
	}

	if err := app.createFramebuffers(); err != nil {
		return err
	}

	return nil
}

func (app *HelloTriangleApplication) cleanupSwapChain() {
	vk.DestroyImageView(app.device, app.depthImageView, nil)
	vk.DestroyImage(app.device, app.depthImage, nil)
	vk.FreeMemory(app.device, app.depthImageMemory, nil)

	for _, framebuffer := range app.swapChainFramebuffers {
		vk.DestroyFramebuffer(app.device, framebuffer, nil)
	}

	for i := range app.swapChainImageViews {
		vk.DestroyImageView(app.device, app.swapChainImageViews[i], nil)
	}

	vk.DestroySwapchain(app.device, app.swapChain, nil)
}

func (app *HelloTriangleApplication) cleanup() {
	app.cleanupSwapChain()

	vk.DestroySampler(app.device, app.textureSampler, nil)
	vk.DestroyImageView(app.device, app.textureImageView, nil)

	vk.DestroyImage(app.device, app.textureImage, nil)
	vk.FreeMemory(app.device, app.textureImageMemory, nil)

	vk.DestroyPipeline(app.device, app.graphicsPipeline, nil)
	vk.DestroyPipelineLayout(app.device, app.pipelineLayout, nil)
	vk.DestroyRenderPass(app.device, app.renderPass, nil)

	for i := range MAX_FRAMES_IN_FLIGHT {
		vk.DestroyBuffer(app.device, app.uniformBuffers[i], nil)
		vk.FreeMemory(app.device, app.uniformBuffersMemories[i], nil)
	}

	vk.DestroyDescriptorPool(app.device, app.descriptorPool, nil)

	vk.DestroyDescriptorSetLayout(app.device, app.descriptorSetLayout, nil)

	vk.DestroyBuffer(app.device, app.indexBuffer, nil)
	vk.FreeMemory(app.device, app.indexBufferMemory, nil)

	vk.DestroyBuffer(app.device, app.vertexBuffer, nil)
	vk.FreeMemory(app.device, app.vertexBufferMemory, nil)

	for i := range MAX_FRAMES_IN_FLIGHT {
		vk.DestroySemaphore(app.device, app.renderFinishedSemaphores[i], nil)
		vk.DestroySemaphore(app.device, app.imageAvailableSemaphores[i], nil)
		vk.DestroyFence(app.device, app.inFlightFences[i], nil)
	}

	vk.DestroyCommandPool(app.device, app.commandPool, nil)

	vk.DestroyDevice(app.device, nil)

	if enableValidationLayers {
		vk.DestroyDebugReportCallback(app.instance, app.debugReportCallback, nil)
	}

	vk.DestroySurface(app.instance, app.surface, nil)
	vk.DestroyInstance(app.instance, nil)

	app.window.Destroy()
	glfw.Terminate()
}

func (app *HelloTriangleApplication) copyBuffer(srcBuffer, dstBuffer vk.Buffer, size vk.DeviceSize) {
	commandBuffer := app.beginSingleTimeCommands()

	copyRegions := []vk.BufferCopy{{Size: size}}
	vk.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, copyRegions)

	app.endSingleTimeCommands(commandBuffer)
}

func (app *HelloTriangleApplication) beginSingleTimeCommands() vk.CommandBuffer {
	allocInfo := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		Level:              vk.CommandBufferLevelPrimary,
		CommandPool:        app.commandPool,
		CommandBufferCount: 1,
	}

	commandBuffers := make([]vk.CommandBuffer, 1)
	vk.AllocateCommandBuffers(app.device, &allocInfo, commandBuffers)

	beginInfo := vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
		Flags: vk.CommandBufferUsageFlags(vk.CommandBufferUsageOneTimeSubmitBit),
	}
	commandBuffer := commandBuffers[0]
	vk.BeginCommandBuffer(commandBuffer, &beginInfo)
	return commandBuffer
}

func (app *HelloTriangleApplication) endSingleTimeCommands(commandBuffer vk.CommandBuffer) {
	vk.EndCommandBuffer(commandBuffer)

	commandBuffers := []vk.CommandBuffer{commandBuffer}

	submitInfos := []vk.SubmitInfo{{
		SType:              vk.StructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    commandBuffers,
	}}

	vk.QueueSubmit(app.graphicsQueue, 1, submitInfos, vk.Fence(vk.NullHandle))
	vk.QueueWaitIdle(app.graphicsQueue)

	vk.FreeCommandBuffers(app.device, app.commandPool, 1, commandBuffers)
}

func (app *HelloTriangleApplication) createBuffer(size vk.DeviceSize, usage vk.BufferUsageFlags, properties vk.MemoryPropertyFlags, buffer *vk.Buffer, bufferMemory *vk.DeviceMemory) error {
	bufferInfo := vk.BufferCreateInfo{
		SType:       vk.StructureTypeBufferCreateInfo,
		Size:        size,
		Usage:       usage,
		SharingMode: vk.SharingModeExclusive,
	}

	if vk.CreateBuffer(app.device, &bufferInfo, nil, buffer) != vk.Success {
		return fmt.Errorf("failed to create buffer")
	}

	var memRequirements vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(app.device, *buffer, &memRequirements)
	memRequirements.Deref()
	memRequirements.Free()

	memoryTypeIndex, err := findMemoryType(memRequirements.MemoryTypeBits, properties, app.physicalDevice)
	if err != nil {
		return err
	}

	allocInfo := vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  memRequirements.Size,
		MemoryTypeIndex: memoryTypeIndex,
	}

	if vk.AllocateMemory(app.device, &allocInfo, nil, bufferMemory) != vk.Success {
		return fmt.Errorf("failed to allocate vertex buffer memory")
	}

	vk.BindBufferMemory(app.device, *buffer, *bufferMemory, 0)
	return nil
}

func (app *HelloTriangleApplication) recordCommandBuffer(commandBuffer vk.CommandBuffer, imageIndex uint32) error {
	beginInfo := vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}

	if vk.BeginCommandBuffer(commandBuffer, &beginInfo) != vk.Success {
		return fmt.Errorf("failed to begin recording command buffer")
	}

	clearValues := make([]vk.ClearValue, 2)
	clearValues[0].SetColor([]float32{0, 0, 0, 1})
	clearValues[1].SetDepthStencil(1, 0)

	renderPassInfo := vk.RenderPassBeginInfo{
		SType:       vk.StructureTypeRenderPassBeginInfo,
		RenderPass:  app.renderPass,
		Framebuffer: app.swapChainFramebuffers[imageIndex],
		RenderArea: vk.Rect2D{
			Offset: vk.Offset2D{
				X: 0, Y: 0,
			},
			Extent: app.swapChainExtent,
		},
		ClearValueCount: uint32(len(clearValues)),
		PClearValues:    clearValues,
	}

	vk.CmdBeginRenderPass(commandBuffer, &renderPassInfo, vk.SubpassContentsInline)
	vk.CmdBindPipeline(commandBuffer, vk.PipelineBindPointGraphics, app.graphicsPipeline)

	viewports := []vk.Viewport{{
		X:        0.0,
		Y:        0.0,
		Width:    float32(app.swapChainExtent.Width),
		Height:   float32(app.swapChainExtent.Height),
		MinDepth: 0.0,
		MaxDepth: 1.0,
	}}
	vk.CmdSetViewport(commandBuffer, 0, 1, viewports)

	scissors := []vk.Rect2D{{
		Offset: vk.Offset2D{X: 0, Y: 0},
		Extent: app.swapChainExtent,
	}}
	vk.CmdSetScissor(commandBuffer, 0, 1, scissors)

	vertexBuffers := []vk.Buffer{app.vertexBuffer}
	offsets := []vk.DeviceSize{0}
	vk.CmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets)

	vk.CmdBindIndexBuffer(commandBuffer, app.indexBuffer, 0, vk.IndexTypeUint16)
	vk.CmdBindDescriptorSets(commandBuffer, vk.PipelineBindPointGraphics, app.pipelineLayout,
		0, 1, []vk.DescriptorSet{app.descriptorSets[app.currentFrame]}, 0, nil)

	vk.CmdDrawIndexed(commandBuffer, uint32(len(indices)), 1, 0, 0, 0)

	vk.CmdEndRenderPass(commandBuffer)

	if vk.EndCommandBuffer(commandBuffer) != vk.Success {
		return fmt.Errorf("failed to record command buffer")
	}
	return nil
}

func hasStencilComponent(format vk.Format) bool {
	return format == vk.FormatD32SfloatS8Uint || format == vk.FormatD24UnormS8Uint
}

func findMemoryType(typeFilter uint32, properties vk.MemoryPropertyFlags, physicalDevice vk.PhysicalDevice) (uint32, error) {
	var memProperties vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties)
	memProperties.Deref()
	memProperties.Free()

	for i, memoryType := range memProperties.MemoryTypes {
		typeBit := uint32(1 << i)
		memoryType.Deref()
		memoryType.Free()

		if (typeFilter&typeBit) != 0 && (memoryType.PropertyFlags&properties) == properties {
			return uint32(i), nil
		}
	}

	return 0, fmt.Errorf("failed to find suitable memory type")
}

func createShaderModule(code []byte, device vk.Device) (vk.ShaderModule, error) {
	createInfo := vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(code)),
		PCode:    bytesToBytecode(code),
	}

	var shaderModule vk.ShaderModule
	err := vk.Error(vk.CreateShaderModule(device, &createInfo, nil, &shaderModule))
	if err != nil {
		return nil, fmt.Errorf("could not create shader module - " + err.Error())
	}
	return shaderModule, nil
}

type swapChainSupportDetails struct {
	capabilities vk.SurfaceCapabilities
	formats      []vk.SurfaceFormat
	presentModes []vk.PresentMode
}

func chooseSwapExtent(capabilities vk.SurfaceCapabilities, window *glfw.Window) vk.Extent2D {
	// capabilities.Deref()
	// capabilities.Free()
	// capabilities.CurrentExtent.Deref()
	// capabilities.CurrentExtent.Free()
	capabilities.MaxImageExtent.Deref()
	defer capabilities.MaxImageExtent.Free()
	capabilities.MinImageExtent.Deref()
	defer capabilities.MinImageExtent.Free()

	// if capabilities.CurrentExtent.Width != vk.MaxUint32 { // ?? imageExtent must be between minImageExtent and maxImageExtent
	// 	return capabilities.CurrentExtent
	// }

	width, height := window.GetFramebufferSize()
	actualExtent := vk.Extent2D{
		Width:  uint32(width),
		Height: uint32(height),
	}

	actualExtent.Width = clamp(actualExtent.Width, capabilities.MinImageExtent.Width, capabilities.MaxImageExtent.Width)
	actualExtent.Height = clamp(actualExtent.Height, capabilities.MinImageExtent.Height, capabilities.MaxImageExtent.Height)
	return actualExtent
}

func chooseSwapPresentMode(availablePresentModes []vk.PresentMode) vk.PresentMode {
	for _, availablePresentMode := range availablePresentModes {
		if availablePresentMode == vk.PresentModeMailbox {
			return availablePresentMode
		}
	}

	return vk.PresentModeFifo
}

func chooseSwapSurfaceFormat(availableFormats []vk.SurfaceFormat) vk.SurfaceFormat {
	if len(availableFormats) < 1 {
		return vk.SurfaceFormat{}
	}

	for _, availableFormat := range availableFormats {
		availableFormat.Deref()
		if availableFormat.Format == vk.FormatB8g8r8a8Srgb && availableFormat.ColorSpace == vk.ColorspaceSrgbNonlinear {
			return availableFormat
		}
		availableFormat.Free()
	}

	return availableFormats[0]
}

func querySwapChainSupport(device vk.PhysicalDevice, surface vk.Surface) swapChainSupportDetails {
	var details swapChainSupportDetails
	vk.GetPhysicalDeviceSurfaceCapabilities(device, surface, &details.capabilities)

	var formatCount uint32
	vk.GetPhysicalDeviceSurfaceFormats(device, surface, &formatCount, nil)
	if formatCount != 0 {
		formats := make([]vk.SurfaceFormat, formatCount)
		vk.GetPhysicalDeviceSurfaceFormats(device, surface, &formatCount, formats)
		details.formats = formats
	}

	var presentModeCount uint32
	vk.GetPhysicalDeviceSurfacePresentModes(device, surface, &presentModeCount, nil)
	if presentModeCount != 0 {
		presentModes := make([]vk.PresentMode, presentModeCount)
		vk.GetPhysicalDeviceSurfacePresentModes(device, surface, &presentModeCount, presentModes)
		details.presentModes = presentModes
	}
	return details
}

func checkDeviceExtensionSupport(device vk.PhysicalDevice) bool {
	var extensionCount uint32
	vk.EnumerateDeviceExtensionProperties(device, "", &extensionCount, nil)
	availableExtensions := make([]vk.ExtensionProperties, extensionCount)
	vk.EnumerateDeviceExtensionProperties(device, "", &extensionCount, availableExtensions)

	requiredExtensions := make(map[string]bool)
	for _, deviceExtension := range deviceExtensions {
		requiredExtensions[strings.Trim(deviceExtension, "\x00")] = true
	}

	for _, availableExtension := range availableExtensions {
		availableExtension.Deref()
		name := vk.ToString(availableExtension.ExtensionName[:])
		_, ok := requiredExtensions[name]
		if ok {
			delete(requiredExtensions, name)
		}
		availableExtension.Free()
	}

	return len(requiredExtensions) == 0
}

func isDeviceSuitable(device vk.PhysicalDevice, surface vk.Surface) bool {
	indices := findQueueFamilies(device, surface)

	extensionsSupported := checkDeviceExtensionSupport(device)

	var swapChainAdequate bool
	if extensionsSupported {
		swapChainSupport := querySwapChainSupport(device, surface)
		swapChainAdequate = len(swapChainSupport.formats) != 0 && len(swapChainSupport.presentModes) != 0
	}

	var supportedFeatures vk.PhysicalDeviceFeatures
	vk.GetPhysicalDeviceFeatures(device, &supportedFeatures)
	supportedFeatures.Deref()
	defer supportedFeatures.Free()

	return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.SamplerAnisotropy.B()
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

func clamp[T uint32](val, min, max T) T {
	if val < min {
		return min
	} else if val > max {
		return max
	}
	return val
}

func bytesToBytecode(b []byte) []uint32 {
	byteCode := make([]uint32, len(b)/4)
	for i := range byteCode {
		byteIndex := i * 4
		byteCode[i] = 0
		byteCode[i] |= uint32(b[byteIndex])
		byteCode[i] |= uint32(b[byteIndex+1]) << 8
		byteCode[i] |= uint32(b[byteIndex+2]) << 16
		byteCode[i] |= uint32(b[byteIndex+3]) << 24
	}

	return byteCode
}

func memcopy(dest unsafe.Pointer, src any) {
	buf := &bytes.Buffer{}
	binary.Write(buf, chooseByteOrder(), src)
	vk.Memcopy(dest, buf.Bytes())
}

func chooseByteOrder() binary.ByteOrder {
	b := uint16(0xff)                      // one byte
	if *(*byte)(unsafe.Pointer(&b)) == 0 { // in big-endian, msb would be stored at the lowest memory address, the 0xff value has been "truncated" to 0x00
		return binary.BigEndian
	} else {
		return binary.LittleEndian
	}
}
