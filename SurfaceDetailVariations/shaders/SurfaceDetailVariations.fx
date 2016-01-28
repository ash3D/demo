//--------------------------------------------------------------------------------------
// File: SurfaceDetailVariations.fx
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
// Includes
//--------------------------------------------------------------------------------------
#include "defines.fxh"
#include "shared.fxh"


//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
static const float3	g_MaterialAmbientColor  = { 0.2, 0.2, 0.2 };	// Material's ambient color
static const float3	g_MaterialDiffuseColor  = { 0.5, 0.5, 0.5 };	// Material's diffuse color
static const float3	g_MaterialSpecularColor = { 0.7, 0.7, 0.7 };	// Material's specular color
static const float3	g_Normal                = { 0.0, 0.0, -1.0 };	// Quad's normal in object space
static const float	g_Shine = 128;
float		g_height_delta_scale;
float		g_max_height;
float		g_grad_threshold;
float		g_max_raytrace_bias;
float		g_linear_search_delta;
int			g_binary_search_steps;
float		g_reflectance;
bool		g_reflections;
bool		g_cast_shadows;
float		g_tex_scale;					// Simple transform in tangent space
float3		g_Eye;							// Eye position in world space
float3		g_WorldLightDir;				// Normalized light's direction in world space
float4		g_LightDiffuse;					// Light's diffuse color
float2		g_normal_height_map_res;		// Resolution of normal height map
int			g_rendering_mode;				// Renderein mode (normal,  final)
int			g_height_diff_samples;			// Technique (number of samples) of height map differentiation
int			g_grad_method;					// Gradients evaluation method

texture2D	g_ColorTexture;					// Color texture for object
texture2D	g_NormalHeightTexture;			// Normal & height texture for object

float4x4	g_mWorld;						// World matrix for object


//--------------------------------------------------------------------------------------
// Texture samplers
//--------------------------------------------------------------------------------------
sampler2D ColorTextureSampler = 
sampler_state
{
    Texture = <g_ColorTexture>;
    MipFilter = LINEAR;
    MinFilter = ANISOTROPIC;
    MagFilter = LINEAR;
    MaxAnisotropy = 16;
    AddressU = MIRROR;
    AddressV = MIRROR;
};


sampler2D NormalHeightTextureSampler = 
sampler_state
{
    Texture = <g_NormalHeightTexture>;
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MaxAnisotropy = 16;
    AddressU = MIRROR;
    AddressV = MIRROR;
};


//--------------------------------------------------------------------------------------
// Vertex shaders output structures
//--------------------------------------------------------------------------------------
struct TRANSFORM_REFLECT_VS_OUTPUT
{
    float4 Position		: POSITION;   // vertex position 
    float2 TextureUV	: TEXCOORD0;  // vertex texture coords 
    float3 WorldPos		: TEXCOORD1;  // vertex position in world space
    float3 Normal		: TEXCOORD2;  // vertex normal in world space
    float3 Reflect		: TEXCOORD3;  // vertex reflection direction in world space
};


struct TRANSFORM_VS_OUTPUT
{
    float4 Position		: POSITION;   // vertex position 
    float2 TextureUV	: TEXCOORD0;  // vertex texture coords 
    float3 WorldPos		: TEXCOORD1;  // vertex position in world space
};


//--------------------------------------------------------------------------------------
// This shader computes standard transform along with world normal and reflection direction
//--------------------------------------------------------------------------------------
TRANSFORM_REFLECT_VS_OUTPUT Transform_Reflect_VS(float3 vPos : POSITION, float2 vTexCoord0 : TEXCOORD0)
{
	TRANSFORM_REFLECT_VS_OUTPUT Output;
	
	float4x4 g_mWorldViewProjection = mul(g_mWorld, g_mViewProjection);
	
	// Transform the position from object space to homogeneous projection space
	Output.Position = mul(float4(vPos, 1), g_mWorldViewProjection);
	
	// Just copy the texture coordinate through
	Output.TextureUV = vTexCoord0;
	
	// Transform the position from object space to world space
	Output.WorldPos = mul(float4(vPos, 1), g_mWorld);
	
	// Transform the normal from object space to world space
	Output.Normal = normalize(mul(g_Normal, (float3x3)g_mWorld)); // normal (world space)
	
	// Find reflection direction in world space
	float3 v = g_Eye - Output.WorldPos;
	Output.Reflect = reflect(-v, Output.Normal);
	
	return Output;
}


//--------------------------------------------------------------------------------------
// This shader computes standard transform
//--------------------------------------------------------------------------------------
TRANSFORM_VS_OUTPUT Transform_VS(float3 vPos : POSITION, float2 vTexCoord0 : TEXCOORD0)
{
	TRANSFORM_VS_OUTPUT Output;
	
	float4x4 g_mWorldViewProjection = mul(g_mWorld, g_mViewProjection);
	
	// Transform the position from object space to homogeneous projection space
	Output.Position = mul(float4(vPos, 1), g_mWorldViewProjection);
	
	// Just copy the texture coordinate through
	Output.TextureUV = vTexCoord0;
	
	// Transform the position from object space to world space
	Output.WorldPos = mul(float4(vPos, 1), g_mWorld);
	
	return Output;
}


//--------------------------------------------------------------------------------------
// Displacement mapping vertex shader
//--------------------------------------------------------------------------------------
TRANSFORM_VS_OUTPUT DisplacementMapping_VS(float3 vPos : POSITION, float2 vTexCoord0 : TEXCOORD0)
{
	TRANSFORM_VS_OUTPUT Output;

	float4x4 g_mWorldViewProjection = mul(g_mWorld, g_mViewProjection);
	
	// Transform the position from object space to homogeneous projection space
	Output.Position = mul(float4(vPos, 1), g_mWorldViewProjection);
	
	// Just copy the texture coordinate through
	Output.TextureUV = vTexCoord0;
	
	// Transform the position from object space to world space
	Output.WorldPos = mul(float4(vPos, 1), g_mWorld);
	
	return Output;
}


//--------------------------------------------------------------------------------------
// Pixel shader output structure
//--------------------------------------------------------------------------------------
struct PS_OUTPUT
{
	float4 RGBColor : COLOR0; // Pixel color
};


//--------------------------------------------------------------------------------------
// These functions returns normal (possible unnormalized) in (+t +b -n) space
//--------------------------------------------------------------------------------------
// Forward declarations 
float3 FetchNormal(uniform sampler2D NormalHeightSampler, in float2 uv);
float3 FetchNormal(uniform sampler2D NormalHeightSampler, in float2 uv, in float2 dx, in float2 dy);

float3 FetchNormal(uniform sampler2D NormalHeightSampler, in float2 uv)
{
	return FetchNormal(NormalHeightSampler, uv, ddx(uv), ddy(uv));
}

float3 FetchNormal(uniform sampler2D NormalHeightSampler, in float2 uv, in float2 dx, in float2 dy)
{
	if(g_height_diff_samples == 0)
	{
		// fetch normal from normal map
		float3 n = tex2Dgrad(NormalHeightSampler, uv, dx, dy).xyz;
		
		// unpack [0..1]->[-1..1]
		n = n * 2.0 - 1.0;
		
		// reconstruct z
		n.z = sqrt(1 - dot(n.xy, n.xy));
		
		return float3(n.x, -n.y, -n.z);
	}
	else
	{
		/*------------------------------------------------------------------------------
		compute normal based on height map in normalized (+t +b +n) space
		U = u / g_tex_scale ( = 2u); dh/dU = 0.5 * dh/du
		V = v / g_tex_scale ( = 2v); dh/dV = 0.5 * dh/dv
		n = (dp/dU x dp/dV) - (dh/dU * dp/dU) - (dh/dV * dp/dV) (*)
		------------------------------------------------------------------------------*/
		// compute delta
//		const float tex_compress = max(length(dx * g_normal_height_map_res), length(dy * g_normal_height_map_res));
//		const float2 delta = clamp((float2)tex_compress, 1, g_normal_height_map_res) / g_normal_height_map_res * g_height_delta_scale;
		const float2 delta = g_height_delta_scale / g_normal_height_map_res;
		
		float3 n;
		// ( * g_tex_scale) for d/du->d/dU, d/dv->d/dV
		// ( / -g_tex_scale) for unnormalized (+T +B -N) -> normalized (+t +b +n) space
		// (-) in formula (*)
		if (g_height_diff_samples == 3)
		{
			float
				height_centre	= tex2Dgrad(NormalHeightSampler, float2(uv.x, uv.y), dx, dy).w,
				height_x		= tex2Dgrad(NormalHeightSampler, float2(uv.x + delta.x, uv.y), dx, dy).w,
				heitht_y		= tex2Dgrad(NormalHeightSampler, float2(uv.x, uv.y + delta.y), dx, dy).w;
			n.x = g_max_height * (height_x - height_centre) / delta;
			n.y = g_max_height * (heitht_y - height_centre) / delta;
		}
		else
			if (g_height_diff_samples == 4)
			{
				float
					height_px = tex2Dgrad(NormalHeightSampler, float2(uv.x + delta.x, uv.y), dx, dy).w,
					height_nx = tex2Dgrad(NormalHeightSampler, float2(uv.x - delta.x, uv.y), dx, dy).w,
					heitht_py = tex2Dgrad(NormalHeightSampler, float2(uv.x, uv.y + delta.y), dx, dy).w,
					heitht_ny = tex2Dgrad(NormalHeightSampler, float2(uv.x, uv.y - delta.y), dx, dy).w;
				n.x = g_max_height * (height_px - height_nx) / (delta.x * 2);
				n.y = g_max_height * (heitht_py - heitht_ny) / (delta.y * 2);
			}
			else // 8 samples
			{
				float
					height_nx_ny = tex2Dgrad(NormalHeightSampler, float2(uv.x - delta.x, uv.y - delta.y), dx, dy).w,
					height_0x_ny = tex2Dgrad(NormalHeightSampler, float2(uv.x, uv.y - delta.y), dx, dy).w,
					height_px_ny = tex2Dgrad(NormalHeightSampler, float2(uv.x + delta.x, uv.y - delta.y), dx, dy).w,
					height_nx_0y = tex2Dgrad(NormalHeightSampler, float2(uv.x - delta.x, uv.y), dx, dy).w,
					height_px_0y = tex2Dgrad(NormalHeightSampler, float2(uv.x + delta.x, uv.y), dx, dy).w,
					height_nx_py = tex2Dgrad(NormalHeightSampler, float2(uv.x - delta.x, uv.y + delta.y), dx, dy).w,
					height_0x_py = tex2Dgrad(NormalHeightSampler, float2(uv.x, uv.y + delta.y), dx, dy).w,
					height_px_py = tex2Dgrad(NormalHeightSampler, float2(uv.x + delta.x, uv.y + delta.y), dx, dy).w;
				n.x = g_max_height * (height_px_ny - height_nx_ny + height_px_py - height_nx_py + 2 * (height_px_0y - height_nx_0y)) / (delta.x * 8);
				n.y = g_max_height * (height_nx_py - height_nx_ny + height_px_py - height_px_ny + 2 * (height_0x_py - height_0x_ny)) / (delta.y * 8);
			}
		n.z = 1;
		
		// return in (+t +b -n) space
		return float3(n.xy, -n.z);
	}
};


//--------------------------------------------------------------------------------------
// This function compute Blinn-Phong lighting
//--------------------------------------------------------------------------------------
float3 BlinnPhong(float3 TexColor, float3 n, float3 v, float3 l, float shadow = 1)
{
//	return v * 0.5 + 0.5;
	float3 h = normalize(v + l);
	float3 color = g_MaterialDiffuseColor * TexColor * saturate(dot(l, n));
	color += g_MaterialSpecularColor * pow(saturate(dot(h, n)), g_Shine);
	color *= g_LightDiffuse;
	color *= shadow;
	color += g_MaterialAmbientColor * TexColor;
	return color;
}


//--------------------------------------------------------------------------------------
// Flat pixel shader
//--------------------------------------------------------------------------------------
PS_OUTPUT Flat_PS(TRANSFORM_REFLECT_VS_OUTPUT In)
{
	PS_OUTPUT Output;
	
	// view and light directions
	float3 v = normalize(g_Eye - In.WorldPos);
	float3 l = normalize(g_WorldLightDir);
	
	float3 n = normalize(In.Normal);
	
	// Lookup texture
	float3 TexColor = tex2D(ColorTextureSampler, In.TextureUV).rgb;
	
	Output.RGBColor.rgb = BlinnPhong(TexColor, n, v, l);
	Output.RGBColor.a = 1.0;
	
	if (g_reflections)
	{
		// Lookup cube map
		float3 reflect_color = texCUBE(CubeMapSampler, In.Reflect);
		
		// Combine with previous color
		Output.RGBColor.rgb = lerp(Output.RGBColor.rgb, reflect_color, g_reflectance);
	}

	return Output;
}


//--------------------------------------------------------------------------------------
// Normal Mapping pixel shader
//--------------------------------------------------------------------------------------
PS_OUTPUT NormalMapping_PS(TRANSFORM_VS_OUTPUT In)
{
	PS_OUTPUT Output;
	
	// view and light directions
	float3 v = normalize(g_Eye - In.WorldPos);
	float3 l = g_WorldLightDir;
	
	// Lookup texture
	float3 TexColor = tex2D(ColorTextureSampler, In.TextureUV).rgb;
	float3 n = FetchNormal(NormalHeightTextureSampler, In.TextureUV);
	
	if (g_rendering_mode == RENDERING_MODE_LOCAL_NORMAL)
		// pack normal [-1..1]->[0..1] and output
		return (PS_OUTPUT)float4((normalize(n) + 1) * .5f, 1);
		
	n = normalize(mul(n, (float3x3)g_mWorld));
	
	if (g_rendering_mode == RENDERING_MODE_WORAL_NORMAL)
		// pack normal [-1..1]->[0..1] and output
		return (PS_OUTPUT)float4((n + 1) * .5f, 1);
	
	Output.RGBColor.rgb = BlinnPhong(TexColor, n, v, l);
	Output.RGBColor.a = 1.0;

	if (g_reflections)
	{
		// find reflection direction in world space
		float3 reflect_dir = reflect(-v, n);
		
		// Lookup cube map
		float3 reflect_color = texCUBE(CubeMapSampler, reflect_dir);
		
		// Combine with previous color
		Output.RGBColor.rgb = lerp(Output.RGBColor.rgb, reflect_color, g_reflectance);
	}

	if (g_rendering_mode == RENDERING_MODE_FINAL)
		return Output;
		
	return (PS_OUTPUT)0;
}


//--------------------------------------------------------------------------------------
// Parallax Mapping 1 pixel shader
//--------------------------------------------------------------------------------------
PS_OUTPUT ParallaxMapping_1_PS(TRANSFORM_VS_OUTPUT In)
{
	PS_OUTPUT Output;

	// view and light directions
	float3 v = normalize(g_Eye - In.WorldPos);
	float3 l = g_WorldLightDir;
	
	// parallax
	float height = (1.0 - tex2D(NormalHeightTextureSampler, In.TextureUV).w) * g_max_height;
	float3 offset = mul((float3x3)g_mWorld, v) * g_tex_scale;
	offset = normalize(float3(offset.xy, -offset.z)) * g_max_height;
	In.TextureUV += height * offset;
	
	// Lookup texture
	float3 TexColor = tex2D(ColorTextureSampler, In.TextureUV).rgb;
	float3 n = FetchNormal(NormalHeightTextureSampler, In.TextureUV);
	
	if (g_rendering_mode == RENDERING_MODE_LOCAL_NORMAL)
		// pack normal [-1..1]->[0..1] and output
		return (PS_OUTPUT)float4((normalize(n) + 1) * .5f, 1);
		
	n = normalize(mul(n, (float3x3)g_mWorld));
	
	if (g_rendering_mode == RENDERING_MODE_WORAL_NORMAL)
		// pack normal [-1..1]->[0..1] and output
		return (PS_OUTPUT)float4((n + 1) * .5f, 1);
	
	Output.RGBColor.rgb = BlinnPhong(TexColor, n, v, l);
	Output.RGBColor.a = 1.0;

	if (g_reflections)
	{
		// find reflection direction in world space
		float3 reflect_dir = reflect(-v, n);
		
		// Lookup cube map
		float3 reflect_color = texCUBE(CubeMapSampler, reflect_dir);
		
		// Combine with previous color
		Output.RGBColor.rgb = lerp(Output.RGBColor.rgb, reflect_color, g_reflectance);
	}

	if (g_rendering_mode == RENDERING_MODE_FINAL)
		return Output;
		
	return (PS_OUTPUT)0;
}


//--------------------------------------------------------------------------------------
// Parallax Mapping 2 pixel shader
//--------------------------------------------------------------------------------------
PS_OUTPUT ParallaxMapping_2_PS(TRANSFORM_VS_OUTPUT In)
{
	PS_OUTPUT Output;

	// view and light directions
	float3 v = normalize(g_Eye - In.WorldPos);
	float3 l = g_WorldLightDir;
	
	// parallax
	float height = (1.0 - tex2D(NormalHeightTextureSampler, In.TextureUV).w) * g_max_height;
	float3 offset = mul((float3x3)g_mWorld, v);
	offset = float3(offset.xy, -offset.z);
	offset *= g_max_height / offset.z;
	In.TextureUV += height * offset;
	
	// Lookup texture
	float3 TexColor = tex2D(ColorTextureSampler, In.TextureUV).rgb;
	float3 n = FetchNormal(NormalHeightTextureSampler, In.TextureUV);
	
	if (g_rendering_mode == RENDERING_MODE_LOCAL_NORMAL)
		// pack normal [-1..1]->[0..1] and output
		return (PS_OUTPUT)float4((normalize(n) + 1) * .5f, 1);
		
	n = normalize(mul(n, (float3x3)g_mWorld));
	
	if (g_rendering_mode == RENDERING_MODE_WORAL_NORMAL)
		// pack normal [-1..1]->[0..1] and output
		return (PS_OUTPUT)float4((n + 1) * .5f, 1);
	
	Output.RGBColor.rgb = BlinnPhong(TexColor, n, v, l);
	Output.RGBColor.a = 1.0;

	if (g_reflections)
	{
		// find reflection direction in world space
		float3 reflect_dir = reflect(-v, n);
		
		// Lookup cube map
		float3 reflect_color = texCUBE(CubeMapSampler, reflect_dir);
		
		// Combine with previous color
		Output.RGBColor.rgb = lerp(Output.RGBColor.rgb, reflect_color, g_reflectance);
	}

	if (g_rendering_mode == RENDERING_MODE_FINAL)
		return Output;
		
	return (PS_OUTPUT)0;
}


//--------------------------------------------------------------------------------------
// Parallax Occlusion Mapping pixel shader
//--------------------------------------------------------------------------------------

// Search intersection point in (+T +B -N) space

static float start_diff;

bool LinearSearch(in float3 dir, out float3 delta, inout float3 offset, in float2 uv, in float2 dx, in float2 dy, in float linear_search_delta, inout int steps)
{
	if (dir.z > 0)
	{
		// search down
		delta = dir / dir.z;
		delta *= g_max_height - offset.z;
	}
	else
	{
		// search up
		delta = dir / -dir.z;
		delta *= offset.z;
	}
	float2 delta_tex = abs(delta.xy);
	const float2 target_delta_tex = linear_search_delta / g_normal_height_map_res;
	float2 delta_scale_xy = target_delta_tex / delta_tex;
	float delta_scale = min(min(delta_scale_xy.x, delta_scale_xy.y), 1);
	delta *= delta_scale;
	
	start_diff = tex2Dgrad(NormalHeightTextureSampler, uv + offset, dx, dy).w * g_max_height - offset.z;
	int cur_step = 0;
	bool no_intersect = true;
	while (offset.z <= g_max_height && offset.z >= 0 && (no_intersect = (tex2Dgrad(NormalHeightTextureSampler, uv + offset, dx, dy).w * g_max_height - offset.z) * start_diff > 0))
	{
		cur_step++;
		offset += delta;
		steps++;
	}
	return no_intersect;
}


void BinarySearch(inout float3 dir, inout float3 delta, inout float3 offset, out float3 offset_back, in float2 uv, in float2 dx, in float2 dy, in int binary_search_steps, inout int steps)
{
	offset_back = offset - delta;
	if (start_diff != 0)
	{
		float delta_len = length(delta);
		for (int cur_step = 0; cur_step < binary_search_steps; cur_step++)
		{
			delta *= 0.5;
			if ((tex2Dgrad(NormalHeightTextureSampler, uv + offset, dx, dy).w * g_max_height - offset.z) * start_diff > 0)
			{
				// outside
				if (delta_len > g_max_raytrace_bias)
					offset_back = offset;
				offset += delta;
			}
			else
				// inside
				offset -= delta;
			delta_len *= 0.5;
		}
		steps += binary_search_steps;
	}
}


PS_OUTPUT ParallaxOcclusionMapping_PS(TRANSFORM_VS_OUTPUT In)
{
	PS_OUTPUT Output;
	
	bool grads_from_base_plane;
	float2
		IntersectUV,	// offset texcoords
		dx, dy;			// corresponding grads
	
	int overall_search_steps = 0;

	// view and light directions
	float3 local_view = In.WorldPos - g_Eye;
	float3 local_light = g_WorldLightDir;
	
	// transform in (+T +B -N) space
	local_view = mul((float3x3)g_mWorld, local_view);
	
	float3 delta;
	
	// offset in (+T +B -N) space
	float3
		offset = 0,
		offset_back;
	
	// compute gradients in base plane
	float2 base_dx, base_dy;
	float4(base_dx, base_dy) = float4(ddx(In.TextureUV), ddy(In.TextureUV));
	
	// search intersection in view direction
	LinearSearch(local_view, delta, offset, In.TextureUV, base_dx, base_dy, g_linear_search_delta, overall_search_steps);
	BinarySearch(local_view, delta, offset, offset_back, In.TextureUV, base_dx, base_dy, g_binary_search_steps, overall_search_steps);
	
	// offset texture coords
	IntersectUV = In.TextureUV + offset;
	
	// transform base offset in world space
	float3 obj_offset = offset / g_tex_scale; // in object space
	float3 world_offset = mul(obj_offset, (float3x3)g_mWorld);
	// Intersection point in world space (used for lighting direction computation for point light sources)
	float3 world_pos = In.WorldPos + world_offset;
//	return (PS_OUTPUT)float4(obj_offset * 0.5 + 0.5, 1);
	
	float shadow_factor;
	
	// search intersection in light direction
	if (g_cast_shadows)
	{
		local_light = mul((float3x3)g_mWorld, local_light); // in (+T +B -N) space
		offset = offset_back;
		shadow_factor = LinearSearch(local_light, delta, offset, In.TextureUV, base_dx, base_dy, g_linear_search_delta, overall_search_steps);
	}
	else
		shadow_factor = 1;

	// lookup texture	
	float3 TexColor;
	float3 n;
	if (g_grad_method == GRAD_METHOD_STD)
		grads_from_base_plane = false;
	else
		if (g_grad_method == GRAD_METHOD_BASE_PLANE)
			grads_from_base_plane = true;
		else
			if (g_grad_method == GRAD_METHOD_HYBRID)
			{
				float4(dx, dy) = float4(ddx(IntersectUV), ddy(IntersectUV));
				grads_from_base_plane = max(dot(dx, dx), dot(dy, dy)) > g_grad_threshold;
			}
			else
				return (PS_OUTPUT)0;
//float4(dx, dy) = float4(ddx(IntersectUV), ddy(IntersectUV));
//if (g_grad_method == GRAD_METHOD_BASE_PLANE || g_grad_method == GRAD_METHOD_HYBRID && max(dot(dx, dx), dot(dy, dy)) > g_grad_threshold)
	if (grads_from_base_plane)
	{
		TexColor = tex2Dgrad(ColorTextureSampler, IntersectUV, base_dx, base_dy).rgb;
		n = FetchNormal(NormalHeightTextureSampler, IntersectUV, base_dx, base_dy).xyz;
	}
	else
	{
		TexColor = tex2D(ColorTextureSampler, IntersectUV).rgb;
		n = FetchNormal(NormalHeightTextureSampler, IntersectUV).xyz;
	}
	
	if (g_height_diff_samples == 0) // normal from normal map
		n.z *= 0.5 / g_max_height;
		
	n = normalize(n);
	
	float3 local_normal = n;
	
	if (g_rendering_mode == RENDERING_MODE_LOCAL_NORMAL)
		// pack normal [-1..1]->[0..1] and output
		return (PS_OUTPUT)float4((local_normal + 1) * .5f, 1);
		
	n = normalize(mul(n, (float3x3)g_mWorld));
	
	if (g_rendering_mode == RENDERING_MODE_WORAL_NORMAL)
		// pack normal [-1..1]->[0..1] and output
		return (PS_OUTPUT)float4((n + 1) * .5f, 1);
	
	float3 v = normalize(g_Eye - In.WorldPos);
	float3 l = g_WorldLightDir;
	
	Output.RGBColor.rgb = BlinnPhong(TexColor, n, v, l, shadow_factor);

	// search intersection in reflection direction
	if (g_reflections)
	{
		offset = offset_back;
		float3 local_reflect = reflect(local_view, local_normal);
		bool cube_map_reflect = LinearSearch(local_reflect, delta, offset, In.TextureUV, base_dx, base_dy, g_linear_search_delta, overall_search_steps);
	
		// Transoform reflection direction in world space
		// Length is meaningless, so it is not required to transform from unnormalized (+T +B -N) space to normalized (+t +b -n) space (object space) (in general case transformation in tangent space required)
		// Normalize for lighting calculations (normalization not required if g_mWorld orthogonal)
		float3 world_reflect = normalize(mul(local_reflect, (float3x3)g_mWorld));
		if (cube_map_reflect)
		{
			float3 reflect_color = texCUBE(CubeMapSampler, world_reflect).rgb;
			Output.RGBColor.rgb = lerp(Output.RGBColor.rgb, reflect_color, g_reflectance);
		}
		else
		{
			BinarySearch(local_reflect, delta, offset, offset_back, In.TextureUV, base_dx, base_dy, g_binary_search_steps, overall_search_steps);
			
			// transform offset in world space
			obj_offset = offset / g_tex_scale; // in object space
			world_offset = mul(obj_offset, (float3x3)g_mWorld);
			// Intersection point in world space (used for lighting direction computation for point light sources)
			world_pos = In.WorldPos + world_offset;

			// offset texture coords
			IntersectUV = In.TextureUV + offset;
			
			if (g_grad_method == GRAD_METHOD_STD)
				grads_from_base_plane = false;
			else
				if (g_grad_method == GRAD_METHOD_BASE_PLANE)
					grads_from_base_plane = true;
				else
					if (g_grad_method == GRAD_METHOD_HYBRID)
					{
						float4(dx, dy) = float4(ddx(IntersectUV), ddy(IntersectUV));
						grads_from_base_plane = max(dot(dx, dx), dot(dy, dy)) > g_grad_threshold;
					}
					else
						return (PS_OUTPUT)0;
//float4(dx, dy) = float4(ddx(IntersectUV), ddy(IntersectUV));
//if (g_grad_method == GRAD_METHOD_BASE_PLANE || g_grad_method == GRAD_METHOD_HYBRID && max(dot(dx, dx), dot(dy, dy)) > g_grad_threshold)
			if (grads_from_base_plane)
			{
				TexColor = tex2Dgrad(ColorTextureSampler, IntersectUV, base_dx, base_dy).rgb;
				n = FetchNormal(NormalHeightTextureSampler, IntersectUV, base_dx, base_dy).xyz;
			}
			else
			{
				TexColor = tex2D(ColorTextureSampler, IntersectUV).rgb;
				n = FetchNormal(NormalHeightTextureSampler, IntersectUV).xyz;
			}
			
			if (g_cast_shadows)
			{
				offset = offset_back;
				shadow_factor = LinearSearch(local_light, delta, offset, In.TextureUV, base_dx, base_dy, g_linear_search_delta, overall_search_steps);
			}
			else
				shadow_factor = 1;
			
			if (g_height_diff_samples == 0) // normal from normal map
				n.z *= 0.5 / g_max_height;
			
			n = normalize(mul(n, (float3x3)g_mWorld));
			float3 reflect_color = BlinnPhong(TexColor, n, -world_reflect, l, shadow_factor);
			Output.RGBColor.rgb = lerp(Output.RGBColor.rgb, reflect_color, g_reflectance);
		}
	}
	
	Output.RGBColor.a = 1.0;
	
//	if (g_rendering_mode == RENDERING_MODE_SEARCH_STEPS)
//		return (PS_OUTPUT)(float(overall_search_steps) / (255 * 6));
	
	if (g_rendering_mode == RENDERING_MODE_FINAL)
		return Output;
		
	return (PS_OUTPUT)0;
}


//--------------------------------------------------------------------------------------
// Displacement mapping pixel shader
//--------------------------------------------------------------------------------------
PS_OUTPUT DisplacementMapping_PS(TRANSFORM_VS_OUTPUT In)
{
	PS_OUTPUT Output;
	
	Output.RGBColor.rgb = In.WorldPos;
	Output.RGBColor.a = 1.0;

	return Output;
}


//--------------------------------------------------------------------------------------
// Techniques
//--------------------------------------------------------------------------------------
technique Flat
{
	pass P0
	{
		VertexShader = compile vs_2_0 Transform_Reflect_VS();
		PixelShader  = compile ps_2_0 Flat_PS();
	}
}

technique NormalMapping
{
	pass P0
	{
		VertexShader = compile vs_3_0 Transform_VS();
		PixelShader  = compile ps_3_0 NormalMapping_PS();
	}
}

technique ParallaxMapping_1
{
	pass P0
	{
		VertexShader = compile vs_3_0 Transform_VS();
		PixelShader  = compile ps_3_0 ParallaxMapping_1_PS();
	}
}

technique ParallaxMapping_2
{
	pass P0
	{
		VertexShader = compile vs_3_0 Transform_VS();
		PixelShader  = compile ps_3_0 ParallaxMapping_2_PS();
	}
}

technique ParallaxOcclusionMapping
{
	pass P0
	{
		VertexShader = compile vs_3_0 Transform_VS();
		PixelShader  = compile ps_3_0 ParallaxOcclusionMapping_PS();
	}
}

technique DisplacementMapping
{
	pass P0
	{
		VertexShader = compile vs_3_0 DisplacementMapping_VS();
		PixelShader  = compile ps_3_0 DisplacementMapping_PS();
	}
}