char simple_vert[] = "\
void main()	\
{	\
	gl_FrontColor = gl_Color;	\
	\
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;	\
}	\
";

char simpleN_vert[] = "\
varying vec3 N;	\
\
void main()	\
{	\
	N = gl_NormalMatrix * gl_Normal;	\
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;	\
}	\
";

char extractAmbient_vert[] = "\
void main()	\
{	\
	gl_TexCoord[0] = gl_MultiTexCoord0;	\
	gl_Position = ftransform();	\
}	\
";

char extractDiffuse_vert[] = "\
void main()	\
{	\
	gl_TexCoord[0] = gl_MultiTexCoord0;	\
	gl_Position = ftransform();	\
}	\
";

char extractSpecular_vert[] = "\
void main()	\
{	\
	gl_TexCoord[0] = gl_MultiTexCoord0;	\
	gl_Position = ftransform();	\
}	\
";

char extractShininess_vert[] = "\
void main()	\
{	\
	gl_TexCoord[0] = gl_MultiTexCoord0;	\
	gl_Position = ftransform();	\
}	\
";

char simple_frag[] = "\
void main()	\
{	\
	gl_FragColor = gl_FrontColor;	\
}	\
";

char extractDepth_frag[] = "\
void main()	\
{	\
	gl_FragColor = vec4(gl_FragCoord.z, gl_FragCoord.z, gl_FragCoord.z, 1);	\
}	\
";

char extractNormal_frag[] = "\
varying vec3 N;	\
\
void main()	\
{	\
	gl_FragColor = vec4(N.x, N.y, N.z, 0);	\
}	\
";

char extractAmbient_frag[] = "\
uniform sampler2D tex;	\
\
void main()	\
{	\
	vec4 texel = texture2D(tex,gl_TexCoord[0].st);	\
	\
	gl_FragColor = gl_FrontMaterial.ambient;	\
	\
	gl_FragColor.w = gl_FragColor.w * texel.w;	\
}	\
";

char extractDiffuse_frag[] = "\
uniform sampler2D tex;	\
\
void main()	\
{	\
	vec4 texel = texture2D(tex,gl_TexCoord[0].st);	\
	\
	gl_FragColor = gl_FrontMaterial.diffuse * texel;	\
}	\
";

char extractSpecular_frag[] = "\
uniform sampler2D tex;	\
\
void main()	\
{	\
	vec4 texel = texture2D(tex,gl_TexCoord[0].st);	\
	\
	gl_FragColor = gl_FrontMaterial.specular;	\
	\
	gl_FragColor.w = gl_FragColor.w * texel.w;	\
}	\
";

char extractShininess_frag[] = "\
uniform sampler2D tex;	\
\
void main()	\
{	\
	vec4 texel = texture2D(tex,gl_TexCoord[0].st);	\
	\
	gl_FragColor = vec4(gl_FrontMaterial.shininess, gl_FrontMaterial.shininess, gl_FrontMaterial.shininess, texel.w);	\
}	\
";





char phong_vert[] = "\
#define NUM_LIGHTS 1	\
varying vec3 normal,lightDir[NUM_LIGHTS],halfVector[NUM_LIGHTS];	\
varying vec4 ambient,diffuse;	\
\
void main()	\
{	\
	vec4 vert = gl_ModelViewMatrix * gl_Vertex;	\
	\
	normal = normalize(gl_NormalMatrix * gl_Normal);	\
	\
	int i;	\
	for(i=0;i<NUM_LIGHTS;i++)	\
	{	\
		lightDir[i] = normalize(vec3(gl_LightSource[i].position-vert));	\
		halfVector[i] = normalize(gl_LightSource[i].halfVector.xyz);	\
	}	\
	\
	gl_TexCoord[0] = gl_MultiTexCoord0;	\
	gl_Position = ftransform();	\
}	\
";

char phong_frag[] = "\
#define NUM_LIGHTS 1   \
varying vec3 normal,lightDir[NUM_LIGHTS],halfVector[NUM_LIGHTS];	\
varying vec4 ambient,diffuse;	\
uniform sampler2D tex;	\
\
void main()	\
{	\
	float NdotL,NdotHV;	\
	vec4 color = gl_LightModel.ambient * gl_FrontMaterial.ambient;	\
	vec4 texel = texture2D(tex,gl_TexCoord[0].st);	\
	\
	int i;	\
	for(int i=0;i<NUM_LIGHTS;i++)	\
	{	\
		NdotL = dot(normal,lightDir[i]);	\
		\
		if (NdotL > 0.0)	\
		{	\
			color += gl_FrontMaterial.diffuse * gl_LightSource[i].diffuse * texel * NdotL + gl_FrontMaterial.ambient * gl_LightSource[i].ambient;	\
			\
			NdotHV = max(dot(normal,halfVector[i]),0.0);	\
			color += gl_FrontMaterial.specular *	\
				gl_LightSource[i].specular *	\
				pow(NdotHV,gl_FrontMaterial.shininess);	\
		}	\
	}	\
	\
	color.w = texel.w;	\
	gl_FragColor = color;	\
}	\
";